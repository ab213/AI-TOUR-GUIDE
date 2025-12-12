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
    print(f"[CFG] Loading config from {os.path.abspath(path)}")
    with open(path) as f:
        cfg = json.load(f)
    print(f"[CFG] Loaded config with keys: {list(cfg.keys())}")
    print(f"[CFG] Models: {[m['name'] for m in cfg.get('models', [])]}")
    print(f"[CFG] Governors: {cfg.get('governors')}")
    print(f"[CFG] Prefill prompts: {list(cfg.get('prefill_prompts', {}).keys())}")
    print(f"[CFG] Decode lengths: {cfg.get('decode_lengths')}")
    print(f"[CFG] Measured runs per config: {cfg.get('measured_runs')}")
    print(f"[CFG] Output dir: {cfg.get('output_dir')}")
    return cfg


def load_tokenizer(hf_path):
    """Load single tokenizer (called only once per model at startup)."""
    repo = hf_path.replace("HF://", "")
    print(f"[TOK]   Loading tokenizer from HF repo: {repo}")
    tok = AutoTokenizer.from_pretrained(repo)
    print(f"[TOK]   Loaded tokenizer vocab size: {tok.vocab_size}")
    return tok


def init_tokenizer_cache(cfg):
    """
    Pre-load all tokenizers once at startup, keyed by hf_path.
    Avoids repeated loading during experiments.
    """
    print("[TOK] Initializing tokenizer cache...")
    cache = {}
    for model_cfg in cfg["models"]:
        hf_path = model_cfg["hf_path"]
        if hf_path not in cache:
            print(f"[TOK] - {model_cfg['name']} ({hf_path}) not in cache; loading...")
            cache[hf_path] = load_tokenizer(hf_path)
        else:
            print(f"[TOK] - {model_cfg['name']} ({hf_path}) already cached; reusing")
    print("[TOK] Tokenizer cache ready.")
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
    print(f"[SYS] Starting background system metrics sampler (interval={interval_s}s)")
    proc = psutil.Process()
    metrics = []
    running = [True]

    def loop():
        while running[0]:
            try:
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
            except Exception as e:
                # Log but do not kill sampler
                print(f"[SYS]   WARN: sampler iteration failed: {type(e).__name__}: {e}")
            time.sleep(interval_s)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return running, metrics


# ============================================================================
# GOVERNOR & POWER UTILITIES
# ============================================================================

def set_governor(mode):
    """Set CPU frequency scaling governor (e.g., 'performance', 'ondemand', 'powersave')."""
    print(f"[GOV] Setting CPU governor to '{mode}'")
    subprocess.run(
        f"echo {mode} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        shell=True, check=False
    )


def load_power_window(t_start, t_end, path="/home/pi/AI-TOUR-GUIDE/benchmarking/results/rd_usb_samples.csv"):
    """
    Load rd-usb power samples within [t_start, t_end] window (Unix timestamps).
    Returns avg_power_w and energy_j via trapezoidal integration over time deltas.
    """
    print(f"[PWR] Loading power window from {path}")
    print(f"[PWR]   Time window: [{t_start:.3f}, {t_end:.3f}]")
    ts, ps = [], []
    try:
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                t = float(row["timestamp"])
                if t_start <= t <= t_end:
                    ts.append(t)
                    ps.append(float(row["power_w"]))
    except FileNotFoundError:
        print(f"[PWR]   File not found; returning zeros.")
        return {"avg_power_w": 0.0, "energy_j": 0.0}

    if not ts:
        print(f"[PWR]   No samples in window; returning zeros.")
        return {"avg_power_w": 0.0, "energy_j": 0.0}

    avg_p = sum(ps) / len(ps)
    energy = 0.0
    for i in range(len(ps) - 1):
        dt = ts[i+1] - ts[i]
        energy += (ps[i] + ps[i+1]) / 2.0 * dt
    if len(ps) == 1:
        energy = ps[0] * (t_end - t_start)  # single sample; assume constant

    print(f"[PWR]   Samples used: {len(ps)}, avg_power_w={avg_p:.3f}, energy_j={energy:.6f}")
    return {"avg_power_w": avg_p, "energy_j": energy}


# ============================================================================
# STREAMING INFERENCE & TOKENIZATION
# ============================================================================

def run_stream_with_ttft(engine, prompt, max_tokens, hf_path):
    """
    Stream chat completion and record: t0 (start), t_first (first token), t_end (completion).
    Returns dict with timestamps and output text for metrics computation.
    """
    print(f"[INF]   Starting streaming inference (max_tokens={max_tokens})")
    t0 = time.time()  # Unix timestamp for alignment with rd-usb
    t_first = None
    out_text = []

    stream = engine.chat.completions.create(
        model=hf_path,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=max_tokens,
    )
    for resp in stream:
        delta = resp.choices[0].delta.content or ""
        if not delta:
            continue
        t_now = time.time()
        if t_first is None:
            t_first = t_now
            print(f"[INF]   First token emitted at t_first={t_first:.3f} (Δ={t_first - t0:.3f}s)")
        out_text.append(delta)

    t_end = time.time()
    print(f"[INF]   Inference complete at t_end={t_end:.3f} (total Δ={t_end - t0:.3f}s)")
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

    print(f"[TOKM]   input_tokens={in_tok}, output_tokens={out_tok}")
    print(f"[TOKM]   prefill_latency_s={prefill_lat:.4f}, decode_latency_s={decode_lat:.4f}")

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
    w = [m for m in metrics if t_start <= m["t"] <= t_end]
    print(f"[SYS]   Window metrics: {len(w)} samples in [{t_start:.3f}, {t_end:.3f}]")
    return w


def summarize_metrics(metrics):
    """Aggregate CPU, memory, thermal, frequency, throttle stats; returns {} if empty."""
    if not metrics:
        print("[SYS]   No metrics for this phase; returning empty summary.")
        return {}
    summary = {
        "avg_cpu": sum(m["cpu"] for m in metrics) / len(metrics),
        "peak_mem_gb": max(m["mem_gb"] for m in metrics),
        "max_temp_c": max(m["temp_c"] for m in metrics),
        "freq_mean_mhz": sum(m["freq_mhz"] for m in metrics) / len(metrics),
        "freq_min_mhz": min(m["freq_mhz"] for m in metrics),
        "freq_max_mhz": max(m["freq_mhz"] for m in metrics),
        "throttled_any": any(m["throttled"] != 0 for m in metrics),
    }
    print(
        f"[SYS]   Summary: avg_cpu={summary['avg_cpu']:.1f}%, "
        f"peak_mem_gb={summary['peak_mem_gb']:.3f}, max_temp_c={summary['max_temp_c']:.1f}, "
        f"freq_mean_mhz={summary['freq_mean_mhz']:.1f}, throttled_any={summary['throttled_any']}"
    )
    return summary


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_one_experiment(engine, tokenizer, prompt, max_tokens, model_name,
                       quantization, activation_precision, governor, prefill_len,
                       decode_len, hf_path):
    """
    Execute single inference run: set governor, sample system metrics in background,
    run streaming inference, align power from rd-usb, and aggregate all metrics.
    Returns single result dict ready for CSV row.
    """
    print("----------------------------------------------------------------")
    print(f"[RUN]   Starting experiment:")
    print(f"[RUN]   Model={model_name}, HF={hf_path}")
    print(f"[RUN]   Governor={governor}, PT={prefill_len}, GT={decode_len}, max_tokens={max_tokens}")
    print(f"[RUN]   Quant={quantization}, ActPrec={activation_precision}")

    # Set governor and stabilize
    if governor:
        set_governor(governor)
        print("[RUN]   Sleeping 1s after governor change...")
        time.sleep(1.0)

    # Start background sampler (Unix timestamps)
    running, metrics = sample_all(interval_s=0.1)
    print("[RUN]   System sampler started.")

    # Warmup
    print("[RUN]   Running warmup inference...")
    try:
        _ = engine.chat.completions.create(
            model=hf_path,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=32,
        )
    except Exception as e:
        print(f"[RUN]   WARN: warmup failed: {type(e).__name__}: {e}")
    print("[RUN]   Warmup done; starting measured run.")

    # Measured run (streaming)
    meas = run_stream_with_ttft(engine, prompt, max_tokens, hf_path)
    running[0] = False  # Stop background sampler
    print("[RUN]   Requested sampler stop flag; waiting short grace period...")
    time.sleep(0.2)

    # Partition metrics into prefill and decode windows
    prefill_metrics = window(metrics, meas["t0"], meas["t_first"])
    decode_metrics = window(metrics, meas["t_first"], meas["t_end"])

    # Load power from rd-usb (Unix timestamps)
    print("[RUN]   Loading power for prefill window...")
    prefill_power = load_power_window(meas["t0"], meas["t_first"])
    print("[RUN]   Loading power for decode window...")
    decode_power = load_power_window(meas["t_first"], meas["t_end"])

    # Token and timing metrics
    tok = token_metrics(tokenizer, prompt, meas)
    print(
        f"[RUN]   TPS: prefill={tok['prefill_tps']:.2f}, "
        f"decode={tok['decode_tps']:.2f}"
    )

    # Per-phase energy and efficiency
    prefill_E_tok_in = (prefill_power["energy_j"] / tok["input_tokens"]
                        if tok["input_tokens"] > 0 else 0.0)
    decode_E_tok_out = (decode_power["energy_j"] / tok["output_tokens"]
                        if tok["output_tokens"] > 0 else 0.0)
    prefill_eff = (tok["prefill_tps"] / prefill_power["avg_power_w"]
                   if prefill_power["avg_power_w"] > 0 else 0.0)
    decode_eff = (tok["decode_tps"] / decode_power["avg_power_w"]
                  if decode_power["avg_power_w"] > 0 else 0.0)

    print(
        f"[RUN]   Energy: prefill={prefill_power['energy_j']:.6f} J "
        f"({prefill_E_tok_in:.6e} J/token in), "
        f"decode={decode_power['energy_j']:.6f} J "
        f"({decode_E_tok_out:.6e} J/token out)"
    )
    print(
        f"[RUN]   Perf/W: prefill={prefill_eff:.6e} tok/s/W, "
        f"decode={decode_eff:.6e} tok/s/W"
    )

    # System metrics summaries
    print("[RUN]   Summarizing prefill system metrics...")
    prefill_sys = summarize_metrics(prefill_metrics)
    print("[RUN]   Summarizing decode system metrics...")
    decode_sys = summarize_metrics(decode_metrics)

    # Aggregate result row
    print("[RUN]   Assembling result row for CSV...")
    result = {
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
        "decode_throttled_any": decode_sys.get("throttled_any"),
    }
    print("[RUN]   Experiment finished.")
    return result


# ============================================================================
# CSV CHECKPOINTING (Resumable Runs)
# ============================================================================

def init_results_csv(output_path):
    """Create CSV header if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"[CSV] Creating new results CSV with header at {output_path}")
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
    else:
        print(f"[CSV] Results CSV already exists at {output_path}; will append.")


def append_result(output_path, result):
    """Append single result row to CSV (atomic)."""
    print(f"[CSV] Appending result row to {output_path}")
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
        print(f"[CSV] No existing CSV at {output_path}; nothing to resume.")
        return completed
    print(f"[CSV] Loading completed runs from {output_path}...")
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                row["model_name"],
                row["quantization"],
                row["activation_precision"],
                row["governor"],
                int(row["prefill_len"]),
                int(row["decode_len"]),
            ))
    print(f"[CSV] Loaded {len(completed)} completed (model, gov, PT, GT) combos.")
    return completed


# ============================================================================
# EXPERIMENT ORCHESTRATION
# ============================================================================

def _cleanup_model_cache(hf_path):
    """Delete cached model to free space between model groups."""
    cache_base = os.path.expanduser("~/test_env/mlc-llm/model_weights/hf/mlc-ai")
    model_cache = os.path.join(cache_base, hf_path)
    print(f"[CLEAN] Requested cleanup for HF path: {hf_path}")
    print(f"[CLEAN] Cache path: {model_cache}")

    if os.path.exists(model_cache):
        try:
            shutil.rmtree(model_cache)
            print(f"[CLEAN]   ✅ Removed cache directory for {hf_path}")
        except Exception as e:
            print(f"[CLEAN]   ⚠️  Could not clean cache for {hf_path}: {e}")
    else:
        print(f"[CLEAN]   No cache directory found for {hf_path}; nothing to do.")


def run_all_experiments(cfg, resume=True, dry_run=False, cleanup_after_model=True):
    """
    Iterate experiment matrix with checkpointing.
    Memory-first order (this runs on a RPi5 with 32gb memory!):
    For each model, run all (governor, prefill, decode, meas_run),
    then delete engine/tokenizer and optionally clean model cache before
    moving to the next model.
    """
    print("================================================================")
    print("[MAIN] Starting run_all_experiments (model-major ordering)")
    print(f"[MAIN]   resume={resume}, dry_run={dry_run}, cleanup_after_model={cleanup_after_model}")
    print("================================================================")

    output_path = os.path.join(cfg["output_dir"], "results.csv")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    init_results_csv(output_path)

    if dry_run:
        print("=" * 70)
        print("[MAIN] DRY RUN MODE: Validating experiment matrix (no inference)")
        print("=" * 70)
        print()

    tokenizer_cache = init_tokenizer_cache(cfg) if not dry_run else {}
    completed = load_completed_runs(output_path) if resume else set()

    total_runs = (len(cfg["governors"]) * len(cfg["models"]) *
                  len(cfg["prefill_prompts"]) * len(cfg["decode_lengths"]) *
                  cfg["measured_runs"])
    print(f"[MAIN] Total (logical) runs in grid: {total_runs}")
    print(f"[MAIN] Unique completed (model, gov, PT, GT): {len(completed)}")
    run_count = 0

    # MODEL OUTER LOOP
    for model_idx, model_cfg in enumerate(cfg["models"], start=1):
        model_name = model_cfg["name"]
        hf_path = model_cfg["hf_path"]
        print(f"\n[MODEL] ========================================================")
        print(f"[MODEL] {model_idx}/{len(cfg['models'])}: {model_name} ({hf_path})")
        print(f"[MODEL] Quant={model_cfg['quantization']}, "
              f"ActPrec={model_cfg['activation_precision']}")
        print(f"[MODEL] Governors: {cfg['governors']}")
        print(f"[MODEL] Prefill lengths: {list(cfg['prefill_prompts'].keys())}")
        print(f"[MODEL] Decode lengths: {cfg['decode_lengths']}")
        print(f"[MODEL] Measured runs per config: {cfg['measured_runs']}")

        if dry_run:
            print(f"[DRY]   (MODEL) Would create engine and run all configs for {model_name}")
            # Still walk config space to log plan
            for gov in cfg["governors"]:
                for prefill_len_str, prompt in cfg["prefill_prompts"].items():
                    prefill_len = int(prefill_len_str)
                    for decode_len in cfg["decode_lengths"]:
                        for meas_run in range(cfg["measured_runs"]):
                            run_count += 1
                            print(f"[DRY]   → {model_name} | {gov} | "
                                  f"{prefill_len}→{decode_len} | run {meas_run+1}/"
                                  f"{cfg['measured_runs']} "
                                  f"(logical {run_count}/{total_runs})")
            # Next model
            continue

        # REAL RUN: create engine + tokenizer once for this model
        print(f"[MODEL] Creating MLCEngine for {hf_path} on CPU...")
        engine = MLCEngine(model=hf_path, device="cpu")
        print(f"[MODEL] Engine ready.")

        tokenizer = tokenizer_cache[hf_path]
        print(f"[MODEL] Tokenizer retrieved from cache.")

        try:
            # INNER LOOPS: governor, prefill, decode, meas_run
            for gov in cfg["governors"]:
                print(f"\n[LOOP] Governor: {gov} (model={model_name})")
                for prefill_len_str, prompt in cfg["prefill_prompts"].items():
                    prefill_len = int(prefill_len_str)
                    print(f"[LOOP]   Prefill length: {prefill_len} (key='{prefill_len_str}')")
                    for decode_len in cfg["decode_lengths"]:
                        print(f"[LOOP]     Decode length: {decode_len}")
                        run_id = (
                                model_name,
                                model_cfg["quantization"],
                                model_cfg["activation_precision"],
                                gov,
                                prefill_len,
                                decode_len
                        )

                        for meas_run in range(cfg["measured_runs"]):
                            run_count += 1
                            print(
                                f"[LOOP]       >> Logical run {run_count}/{total_runs} "
                                f"(meas_run={meas_run+1}/{cfg['measured_runs']})"
                            )

                            # Resume / skip logic per (model, gov, PT, GT)
                            if run_id in completed:
                                if meas_run == 0:
                                    print(
                                        f"[SKIP] {model_name} | {gov} | "
                                        f"{prefill_len}→{decode_len} already completed "
                                        f"(at least one run); skipping remaining repeats."
                                    )
                                else:
                                    print(
                                        f"[SKIP]   Repeated run {meas_run+1} skipped "
                                        f"for completed combo {run_id}"
                                    )
                                continue

                            # Run experiment
                            try:
                                result = run_one_experiment(
                                    engine=engine,
                                    tokenizer=tokenizer,
                                    prompt=prompt,
                                    max_tokens=decode_len,
                                    model_name=model_name,
                                    quantization=model_cfg["quantization"],
                                    activation_precision=model_cfg["activation_precision"],
                                    governor=gov,
                                    prefill_len=prefill_len,
                                    decode_len=decode_len,
                                    hf_path=hf_path,
                                )
                                append_result(output_path, result)
                                print(
                                    f"[OK]  [{run_count}/{total_runs}] "
                                    f"{model_name} | {gov} | {prefill_len}→{decode_len} "
                                    f"(run {meas_run+1}/{cfg['measured_runs']}) COMPLETE"
                                )
                            except Exception as e:
                                print(
                                    f"[ERR] [{run_count}/{total_runs}] "
                                    f"{model_name} | {gov} | {prefill_len}→{decode_len} "
                                    f"(run {meas_run+1}/{cfg['measured_runs']}) FAILED"
                                )
                                print(f"[ERR]   {type(e).__name__}: {e}")
                                # Continue to next config, but keep engine alive
                                continue
        finally:
            # ALWAYS free this model's resources before moving on
            print(f"\n[MODEL] Cleaning up in-memory objects for {model_name}...")
            try:
                del engine
            except NameError:
                pass
            try:
                del tokenizer
            except NameError:
                pass

            import gc
            gc.collect()
            print("[MODEL] Python GC collected; engine/tokenizer deleted.")

            if cleanup_after_model:
                print(f"[MODEL] cleanup_after_model=True; cleaning on-disk cache for {hf_path}")
                _cleanup_model_cache(hf_path)
            else:
                print("[MODEL] cleanup_after_model=False; skipping on-disk cache cleanup.")

        print(f"[MODEL] Finished all configs for {model_name}. Moving to next model.")

    if dry_run:
        print()
        print("=" * 70)
        print(f"[MAIN] DRY RUN SUMMARY")
        print(f"[MAIN]   Total logical runs: {total_runs}")
        print(f"[MAIN]   Governors: {cfg['governors']}")
        print(f"[MAIN]   Models: {len(cfg['models'])}")
        print(f"[MAIN]   Prefill lengths: {list(cfg['prefill_prompts'].keys())}")
        print(f"[MAIN]   Decode lengths: {cfg['decode_lengths']}")
        print(f"[MAIN]   Measured runs per config: {cfg['measured_runs']}")
        print("=" * 70)
    else:
        print(f"\n[MAIN] ✓ All models processed. Results saved to {output_path}")


if __name__ == "__main__":
    print("[ENTRY] benchmark_orchestrator.py starting...")
    cfg = load_config("benchmark_config.json")
    dry_run = "--dry-run" in sys.argv
    print(f"[ENTRY] CLI args: {sys.argv}")
    print(f"[ENTRY] dry_run={dry_run}")

    # Env flags from shell script
    no_resume = os.getenv("NO_RESUME", "0") == "1"
    resume = not no_resume
    cleanup_after_model = os.getenv("CLEANUP_AFTER_MODEL", "0") == "1"

    print(f"[ENTRY] Env flags: NO_RESUME={no_resume}, CLEANUP_AFTER_MODEL={cleanup_after_model}")
    print(f"[ENTRY] Effective flags: resume={resume}, cleanup_after_model={cleanup_after_model}")

    run_all_experiments(cfg, resume=resume, dry_run=dry_run, cleanup_after_model=cleanup_after_model)

