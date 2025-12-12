#!/usr/bin/env python
import csv
import json
import os
import time
from collections import defaultdict
from statistics import mean

CONFIG_PATH = "benchmark_config.json"
RESULTS_PATH = "results/results.csv"

def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)

def load_results(path=RESULTS_PATH):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_full_grid(cfg):
    """Return list of all logical configs that orchestrator will run."""
    grid = []
    for model in cfg["models"]:
        for gov in cfg["governors"]:
            for prefill_len_str in cfg["prefill_prompts"].keys():
                prefill_len = int(prefill_len_str)
                for decode_len in cfg["decode_lengths"]:
                    grid.append((
                        model["name"],
                        model["quantization"],
                        model["activation_precision"],
                        gov,
                        prefill_len,
                        decode_len,
                    ))
    return grid

def summarize_completed(cfg, rows):
    """
    Determine which (model, quant, actprec, gov, PT, GT) configs
    have at least one completed run, and estimate per-config runtime
    from measured latency.
    """
    # Map from config key -> list of estimated runtimes (seconds)
    per_config_durations = defaultdict(list)

    for row in rows:
        try:
            key = (
                row["model_name"],
                row["quantization"],
                row["activation_precision"],
                row["governor"],
                int(row["prefill_len"]),
                int(row["decode_len"]),
            )
        except KeyError:
            # Old CSVs without new fields; skip
            continue

        # Approximate wall time of a single measured run:
        # warmup (~few sec) + prefill + decode; be conservative with a small fudge factor
        try:
            prefill = float(row["prefill_latency_s"])
            decode = float(row["decode_latency_s"])
        except (KeyError, ValueError):
            continue

        est = prefill + decode + 5.0  # 5s overhead (governor switch, warmup, etc.)
        per_config_durations[key].append(est)

    # For each config, store avg duration of a single measured run
    config_avg = {k: mean(v) for k, v in per_config_durations.items()}
    return config_avg

def humanize_seconds(s):
    s = int(s)
    if s <= 0:
        return "0s"
    mins, sec = divmod(s, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hrs > 0:
        parts.append(f"{hrs}h")
    if mins > 0:
        parts.append(f"{mins}m")
    if sec > 0 or not parts:
        parts.append(f"{sec}s")
    return " ".join(parts)

def main():
    cfg = load_config()
    rows = load_results()

    measured_runs = cfg["measured_runs"]
    grid = build_full_grid(cfg)
    total_logical_runs = len(grid) * measured_runs

    # Completed configs and per-config avg run time (from results.csv)
    config_avg = summarize_completed(cfg, rows)
    completed_config_keys = set(config_avg.keys())

    # Count completed logical runs directly from CSV
    completed_logical_runs = len(rows)

    # Remaining configs = those from grid that don't appear in config_avg
    remaining_configs = [g for g in grid if g not in completed_config_keys]
    remaining_config_count = len(remaining_configs)

    # For configs that already have some runs, how many repeats remain?
    # (e.g., if measured_runs=3 and there is 1 row, remaining repeats=2)
    partial_repeats = 0
    for key in grid:
        # Rows matching this key
        count_for_key = sum(
            1 for r in rows
            if r.get("model_name") == key[0]
            and r.get("quantization") == key[1]
            and r.get("activation_precision") == key[2]
            and r.get("governor") == key[3]
            and int(r.get("prefill_len", 0)) == key[4]
            and int(r.get("decode_len", 0)) == key[5]
        )
        if 0 < count_for_key < measured_runs:
            partial_repeats += (measured_runs - count_for_key)

    # Estimate per-logical-run time.
    # Use all existing rows' (prefill+decode) as samples.
    per_run_durations = []
    for r in rows:
        try:
            prefill = float(r["prefill_latency_s"])
            decode = float(r["decode_latency_s"])
        except (KeyError, ValueError):
            continue
        per_run_durations.append(prefill + decode + 5.0)

    if per_run_durations:
        global_avg_run = mean(per_run_durations)
    else:
        global_avg_run = 60.0  # fallback guess, 1 minute per run

    # Estimate time for:
    # - configs with no runs yet: measured_runs * global_avg_run per config
    # - partial repeats: each remaining repeat ≈ global_avg_run
    est_time_remaining_s = remaining_config_count * measured_runs * global_avg_run + \
                           partial_repeats * global_avg_run

    print("=== Benchmark Progress Estimate ===")
    print(f"Config file:    {os.path.abspath(CONFIG_PATH)}")
    print(f"Results file:   {os.path.abspath(RESULTS_PATH)}")
    print()
    print(f"Measured runs per config:      {measured_runs}")
    print(f"Total logical runs (grid):     {total_logical_runs}")
    print(f"Completed logical runs (rows): {completed_logical_runs}")
    print(f"Unique configs in grid:        {len(grid)}")
    print(f"Configs with >=1 run:          {len(set(config_avg.keys()) & set(grid))}")
    print(f"Configs with 0 runs:           {remaining_config_count}")
    print(f"Configs with partial repeats:  {partial_repeats}")
    print()
    print(f"Estimated avg wall time/run:   {global_avg_run:.1f} s "
          f"({humanize_seconds(global_avg_run)})")
    print(f"Estimated remaining wall time: {est_time_remaining_s:.1f} s "
          f"({humanize_seconds(est_time_remaining_s)})")

    # Optional ETC based on current time
    etc = time.time() + est_time_remaining_s
    print(f"Estimated completion time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(etc))}")

if __name__ == "__main__":
    main()

