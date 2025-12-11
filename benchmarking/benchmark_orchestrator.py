import time
import csv
import json
import subprocess
import psutil
import threading
import re
import os
import sys
import shutil
from mlc_llm import MLCEngine
from transformers import AutoTokenizer


# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

def load_config(path="benchmark_config.json"):
    """Load experiment config from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_tokenizer(hf_path):
    """Load single tokenizer (called only once per model at startup)."""
    repo = hf_path.replace("HF://", "")
    return AutoTokenizer.from_pretrained(repo)


def init_tokenizer_cache(cfg):
    """
    Pre-load all tokenizers once at startup, keyed by hf_path.
    Avoids repeated loading during experiments.
    """
    cache = {}
    for model_cfg in cfg["models"]:
        hf_path = model_cfg["hf_path"]
        if hf_path not in cache:
            print(f"Loading tokenizer for {model_cfg['name']}...")
            cache[hf_path] = load_tokenizer(hf_path)
    return cache


# ============================================================================
# SYSTEM METRICS SAMPLER (Background Thread)
# ============================================================================

def sample_all(interval_s=0.1):
    """
    Background sampler thread for system metrics (CPU, mem, temp, freq, throttle, volts).
    Returns (running_flag, metrics_list). Caller must set running[0] = False to stop.
    Uses a mutable list for running flag to allow thread to see updates without locks.
    """
    proc = psutil.Process()
    metrics = []
    running = [True]

    def loop():
        while running[0]:
            cpu = psutil.cpu_percent(interval=None)  # non-blocking
            mem_gb = proc.memory_info().rss / (1024**3)
            
            temp_out = subprocess.check_output("vcgencmd measure_temp", shell=True).decode()
            temp_c = float(re.search(r'([\d.]+)', temp_out).group(1))
            
            freq_out = subprocess.check_output("vcgencmd measure_clock arm", shell=True).decode()
            freq_mhz = int(re.search(r'=(\d+)', freq_out).group(1)) / 1_000_000
            
            thr_out = subprocess.check_output("vcgencmd get_throttled", shell=True).decode()
            throttled = int(re.search(r'0x([0-9a-fA-F]+)', thr_out).group(1), 16)
            
            volts_out = subprocess.check_output("vcgencmd measure_volts", shell=True).decode()
            volts = float(re.search(r'([\d.]+)', volts_out).group(1))

            metrics.append({
                "t": time.time(),  # Unix timestamp for alignment with rd-usb
                "cpu": cpu,
                "mem_gb": mem_gb,
                "temp_c": temp_c,
                "freq_mhz": freq_mhz,
                "throttled": throttled,
                "volts": volts,
            })
            time.sleep(interval_s)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return running, metrics


# ============================================================================
# GOVERNOR & POWER UTILITIES
# ============================================================================

def set_governor(mode):
    """Set CPU frequency scaling governor (e.g., 'performance', 'ondemand', 'powersave')."""
    subprocess.run(
        f"echo {mode} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        shell=True, check=False
    )


def load_power_window(t_start, t_end, path="/tmp/rd_usb_samples.csv"):
    """
    Load rd-usb power samples within [t_start, t_end] window (Unix timestamps).
    Returns avg_power_w and energy_j via trapezoidal integration over time deltas.
    """
    ts, ps = [], []
    try:
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                t = float(row["timestamp"])
                if t_start <= t <= t_end:
                    ts.append(t)
                    ps.append(float(row["power"]))
    except FileNotFoundError:
        return {"avg_power_w": 0.0, "energy_j": 0.0}
    
    if not ts:
        return {"avg_power_w": 0.0, "energy_j": 0.0}
    
    avg_p = sum(ps) / len(ps)
    # Trapezoidal integration: ∑((p_i + p_{i+1})/2 * Δt_i) where Δt_i = t_{i+1} - t_i
    energy = 0.0
    for i in range(len(ps) - 1):
        dt = ts[i+1] - ts[i]
        energy += (ps[i] + ps[i+1]) / 2.0 * dt
    if len(ps) == 1:
        energy = ps[0] * (t_end - t_start)  # single sample; assume constant
    
    return {"avg_power_w": avg_p, "energy_j": energy}


# ============================================================================
# STREAMING INFERENCE & TOKENIZATION
# ============================================================================

def run_stream_with_ttft(engine, prompt, max_new_tokens):
    """
    Stream chat completion and record: t0 (start), t_first (first token), t_end (completion).
    Returns dict with timestamps and output text for metrics computation.
    """
    t0 = time.time()  # Unix timestamp for alignment with rd-usb
    t_first = None
    out_text = []

    stream = engine.chat.completions.create(
        model=engine.model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=max_new_tokens,
    )
    for resp in stream:
        delta = resp.choices[0].delta.content or ""
        if not delta:
            continue
        t_now = time.time()
        if t_first is None:
            t_first = t_now
        out_text.append(delta)

    t_end = time.time()
    return {
        "t0": t0,
        "t_first": t_first,
        "t_end": t_end,
        "output": "".join(out_text),
    }


def token_metrics(tokenizer, prompt, meas):
    """
    Compute token counts and latency/throughput per phase (prefill, decode).
    Tokenizes output only (prompt is known from config).
    """
    in_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
    out_tok = len(tokenizer.encode(meas["output"], add_special_tokens=False))
    prefill_lat = meas["t_first"] - meas["t0"]
    decode_lat = meas["t_end"] - meas["t_first"]
    
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "prefill_latency_s": prefill_lat,
        "decode_latency_s": decode_lat,
        "prefill_tps": in_tok / prefill_lat if prefill_lat > 0 else 0.0,
        "decode_tps": out_tok / decode_lat if decode_lat > 0 else 0.0,
    }


# ============================================================================
# SYSTEM METRICS AGGREGATION
# ============================================================================

def window(metrics, t_start, t_end):
    """Filter metrics to time window [t_start, t_end] (Unix timestamps)."""
    return [m for m in metrics if t_start <= m["t"] <= t_end]


def summarize_metrics(metrics):
    """Aggregate CPU, memory, thermal, frequency, throttle stats; returns {} if empty."""
    if not metrics:
        return {}
    return {
        "avg_cpu": sum(m["cpu"] for m in metrics) / len(metrics),
        "peak_mem_gb": max(m["mem_gb"] for m in metrics),
        "max_temp_c": max(m["temp_c"] for m in metrics),
        "freq_mean_mhz": sum(m["freq_mhz"] for m in metrics) / len(metrics),
        "freq_min_mhz": min(m["freq_mhz"] for m in metrics),
        "freq_max_mhz": max(m["freq_mhz"] for m in metrics),
        "throttled_any": any(m["throttled"] != 0 for m in metrics),
    }


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_one_experiment(engine, tokenizer, prompt, max_new_tokens, model_name,
                       quantization, activation_precision, governor, prefill_len,
                       decode_len):
    """
    Execute single inference run: set governor, sample system metrics in background,
    run streaming inference, align power from rd-usb, and aggregate all metrics.
    Returns single result dict ready for CSV row.
    """
    # Set governor and stabilize
    if governor:
        set_governor(governor)
        time.sleep(1.0)

    # Start background sampler (Unix timestamps)
    running, metrics = sample_all(interval_s=0.1)

    # Warmup
    for _ in range(1):  # Fixed warmup (1 run)
        _ = engine.chat.completions.create(
            model=engine.model,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=32,
        )

    # Measured run (streaming)
    meas = run_stream_with_ttft(engine, prompt, max_new_tokens)
    running[0] = False  # Stop background sampler

    # Partition metrics into prefill and decode windows
    prefill_metrics = window(metrics, meas["t0"], meas["t_first"])
    decode_metrics = window(metrics, meas["t_first"], meas["t_end"])

    # Load power from rd-usb (Unix timestamps)
    prefill_power = load_power_window(meas["t0"], meas["t_first"])
    decode_power = load_power_window(meas["t_first"], meas["t_end"])

    # Token and timing metrics
    tok = token_metrics(tokenizer, prompt, meas)

    # Per-phase energy and efficiency
    prefill_E_tok_in = (prefill_power["energy_j"] / tok["input_tokens"]
                        if tok["input_tokens"] > 0 else 0.0)
    decode_E_tok_out = (decode_power["energy_j"] / tok["output_tokens"]
                        if tok["output_tokens"] > 0 else 0.0)
    prefill_eff = (tok["prefill_tps"] / prefill_power["avg_power_w"]
                   if prefill_power["avg_power_w"] > 0 else 0.0)
    decode_eff = (tok["decode_tps"] / decode_power["avg_power_w"]
                  if decode_power["avg_power_w"] > 0 else 0.0)

    # System metrics summaries
    prefill_sys = summarize_metrics(prefill_metrics)
    decode_sys = summarize_metrics(decode_metrics)

    # Aggregate result row
    return {
        "model_name": model_name,
        "quantization": quantization,
        "activation_precision": activation_precision,
        "governor": governor,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        # Token & timing
        "input_tokens": tok["input_tokens"],
        "output_tokens": tok["output_tokens"],
        "prefill_latency_s": tok["prefill_latency_s"],
        "decode_latency_s": tok["decode_latency_s"],
        "prefill_tps": tok["prefill_tps"],
        "decode_tps": tok["decode_tps"],
        # Power & energy
        "prefill_avg_power_w": prefill_power["avg_power_w"],
        "decode_avg_power_w": decode_power["avg_power_w"],
        "prefill_energy_j": prefill_power["energy_j"],
        "decode_energy_j": decode_power["energy_j"],
        "prefill_energy_per_token_in": prefill_E_tok_in,
        "decode_energy_per_token_out": decode_E_tok_out,
        "prefill_tokens_per_s_per_w": prefill_eff,
        "decode_tokens_per_s_per_w": decode_eff,
        # Prefill system
        "prefill_avg_cpu": prefill_sys.get("avg_cpu"),
        "prefill_peak_mem_gb": prefill_sys.get("peak_mem_gb"),
        "prefill_max_temp_c": prefill_sys.get("max_temp_c"),
        "prefill_freq_mean_mhz": prefill_sys.get("freq_mean_mhz"),
        "prefill_freq_min_mhz": prefill_sys.get("freq_min_mhz"),
        "prefill_freq_max_mhz": prefill_sys.get("freq_max_mhz"),
        "prefill_throttled_any": prefill_sys.get("throttled_any"),
        # Decode system
        "decode_avg_cpu": decode_sys.get("avg_cpu"),
        "decode_peak_mem_gb": decode_sys.get("peak_mem_gb"),
        "decode_max_temp_c": decode_sys.get("max_temp_c"),
        "decode_freq_mean_mhz": decode_sys.get("freq_mean_mhz"),
        "decode_freq_min_mhz": decode_sys.get("freq_min_mhz"),
        "decode_freq_max_mhz": decode_sys.get("freq_max_mhz"),
        "decode_throttled_any": decode_sys.get("decode_throttled_any"),
    }


# ============================================================================
# CSV CHECKPOINTING (Resumable Runs)
# ============================================================================

def init_results_csv(output_path):
    """Create CSV header if it doesn't exist."""
    if not os.path.exists(output_path):
        fieldnames = [
            "model_name", "quantization", "activation_precision", "governor",
            "prefill_len", "decode_len",
            "input_tokens", "output_tokens",
            "prefill_latency_s", "decode_latency_s", "prefill_tps", "decode_tps",
            "prefill_avg_power_w", "decode_avg_power_w",
            "prefill_energy_j", "decode_energy_j",
            "prefill_energy_per_token_in", "decode_energy_per_token_out",
            "prefill_tokens_per_s_per_w", "decode_tokens_per_s_per_w",
            "prefill_avg_cpu", "prefill_peak_mem_gb", "prefill_max_temp_c",
            "prefill_freq_mean_mhz", "prefill_freq_min_mhz", "prefill_freq_max_mhz",
            "prefill_throttled_any",
            "decode_avg_cpu", "decode_peak_mem_gb", "decode_max_temp_c",
            "decode_freq_mean_mhz", "decode_freq_min_mhz", "decode_freq_max_mhz",
            "decode_throttled_any",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_result(output_path, result):
    """Append single result row to CSV (atomic)."""
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writerow(result)


def load_completed_runs(output_path):
    """
    Return set of (model_name, gov, prefill_len, decode_len) tuples already completed.
    Used to skip/resume.
    """
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                row["model_name"],
                row["governor"],
                int(row["prefill_len"]),
                int(row["decode_len"]),
            ))
    return completed


# ============================================================================
# EXPERIMENT ORCHESTRATION
# ============================================================================

def _cleanup_model_cache(hf_path):
    """Delete cached model to free space between model groups."""
    cache_base = os.path.expanduser("~/morph_env/mlc-llm/model_weights/hf/mlc-ai")
    model_name = hf_path.split("/")[-1].replace("-MLC", "")
    model_cache = os.path.join(cache_base, model_name)
    
    if os.path.exists(model_cache):
        try:
            shutil.rmtree(model_cache)
            print(f"  🗑️  Cleaned up cache for {model_name}")
        except Exception as e:
            print(f"  ⚠️  Could not clean up {model_name}: {e}")


def run_all_experiments(cfg, resume=True, dry_run=False, cleanup_after_model=False):
    """
    Iterate experiment matrix with checkpointing. Skip completed runs if resume=True.
    If dry_run=True, skip actual inference; just validate structure and log plan.
    """
    output_path = os.path.join(cfg["output_dir"], "results.csv")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    init_results_csv(output_path)
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE: Validating experiment matrix (no inference)")
        print("=" * 70)
        print()
    
    tokenizer_cache = init_tokenizer_cache(cfg) if not dry_run else {}
    completed = load_completed_runs(output_path) if resume else set()
    total_runs = (len(cfg["governors"]) * len(cfg["models"]) * 
                  len(cfg["prefill_prompts"]) * len(cfg["decode_lengths"]) *
                  cfg["measured_runs"])
    run_count = 0

    for gov in cfg["governors"]:
        last_model = None
        for model_cfg in cfg["models"]:
            # Clean up previous model if requested (frees space between models)
            if cleanup_after_model and last_model and last_model != model_cfg["hf_path"]:
                _cleanup_model_cache(last_model)

            if dry_run:
                print(f"[DRY] Model: {model_cfg['name']} ({model_cfg['quantization']})")
                continue
            
            engine = MLCEngine(model=model_cfg["hf_path"], device="cpu")
            last_model = model_cfg["hf_path"]
            tokenizer = tokenizer_cache[model_cfg["hf_path"]]

            for prefill_len_str in cfg["prefill_prompts"].keys():
                prefill_len = int(prefill_len_str)
                prompt = cfg["prefill_prompts"][prefill_len_str]
                
                for decode_len in cfg["decode_lengths"]:
                    for meas_run in range(cfg["measured_runs"]):
                        if dry_run:
                            print(f"  → {gov} | {prefill_len}→{decode_len} tokens | run {meas_run+1}/{cfg['measured_runs']}")
                            run_count += 1
                            continue
                        
                        run_id = (model_cfg["name"], gov, prefill_len, decode_len)
                        if run_id in completed:
                            if meas_run == 0:
                                print(f"⊘ [{run_count}/{total_runs}] Skipping completed: {model_cfg['name']} | {gov} | {prefill_len}→{decode_len}")
                            run_count += 1
                            continue

                        run_count += 1

                        try:
                            result = run_one_experiment(
                                engine=engine,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_new_tokens=decode_len,
                                model_name=model_cfg["name"],
                                quantization=model_cfg["quantization"],
                                activation_precision=model_cfg["activation_precision"],
                                governor=gov,
                                prefill_len=prefill_len,
                                decode_len=decode_len,
                            )
                            append_result(output_path, result)
                            print(f"✓ [{run_count}/{total_runs}] {model_cfg['name']} | {gov} | {prefill_len}→{decode_len} (run {meas_run+1}/{cfg['measured_runs']})")
                        except Exception as e:
                            print(f"✗ [{run_count}/{total_runs}] {model_cfg['name']} | {gov} | {prefill_len}→{decode_len} (run {meas_run+1}/{cfg['measured_runs']})")
                            print(f"  Error: {type(e).__name__}: {e}")
                            continue

    if dry_run:
        print()
        print("=" * 70)
        print(f"DRY RUN: {total_runs} experiments planned")
        print(f"Governors: {cfg['governors']}")
        print(f"Models: {len(cfg['models'])}")
        print(f"Prefill lengths: {list(cfg['prefill_prompts'].keys())}")
        print(f"Decode lengths: {cfg['decode_lengths']}")
        print(f"Measured runs per config: {cfg['measured_runs']}")
        print("=" * 70)
    else:
        print(f"\n✓ Experiment complete. Results saved to {output_path}")

if __name__ == "__main__":
    cfg = load_config("benchmark_config.json")
    dry_run = "--dry-run" in sys.argv
    cleanup_after_model = "--cleanup-after-model" in sys.argv
    
    if "--precache" in sys.argv:
        precache_models(cfg)
        sys.exit(0)
    
    run_all_experiments(cfg, resume=True, dry_run=dry_run, cleanup_after_model=cleanup_after_model)

