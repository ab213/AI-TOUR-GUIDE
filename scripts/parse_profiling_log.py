#!/usr/bin/env python3
"""
Parse profiling.log to extract completed experiment results.

Usage:
    python scripts/parse_profiling_log.py --log-file profiling.log --output profiling_results/log_results.csv
"""

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime


EXPERIMENT_RE = re.compile(
    r"^Experiment:\s+(?P<model>.+?)\s+\|\s+(?P<quantization>.+?)\s+\|\s+(?P<activation_precision>.+?)\s+\|\s+Prefill:(?P<prefill>\d+)\s+\|\s+Decode:(?P<decode>\d+)",
    re.MULTILINE,
)
RUN_RE = re.compile(
    r"Run\s+(?P<idx>\d+)/(?P<total>\d+)\s+-\s+Latency:\s+(?P<prefill_latency>[\d.]+)s prefill,\s+(?P<decode_latency>[\d.]+)s decode"
)


def parse_log(log_text: str):
    records = []
    blocks = log_text.split("Experiment:")
    for block in blocks[1:]:
        header_line = "Experiment:" + block.splitlines()[0]
        experiment_match = EXPERIMENT_RE.search(header_line)
        if not experiment_match:
            continue
        info = experiment_match.groupdict()
        model = info["model"].strip()
        quantization = info["quantization"].strip()
        activation_precision = info["activation_precision"].strip()
        prefill_len = int(info["prefill"])
        decode_len = int(info["decode"])

        run_matches = RUN_RE.finditer(block)
        for run_match in run_matches:
            run_info = run_match.groupdict()
            prefill_latency = float(run_info["prefill_latency"])
            decode_latency = float(run_info["decode_latency"])

            tokens_per_sec = decode_len / decode_latency if decode_latency > 0 else 0.0
            timestamp = datetime.now().isoformat()
            record = {
                "model": model,
                "quantization": quantization,
                "activation_precision": activation_precision,
                "prefill_len": prefill_len,
                "decode_len": decode_len,
                "prefill_latency": prefill_latency,
                "decode_latency": decode_latency,
                "tokens_per_sec": tokens_per_sec,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "memory_mb": -1.0,
                "cpu_pct": -1.0,
                "temp_c": -1.0,
                "power_w": -1.0,
                "timestamp": timestamp,
                "source": "log",
            }
            records.append(record)
    return records


def write_csv(records, output_path: Path):
    if not records:
        print("No records parsed from log.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "quantization",
        "activation_precision",
        "prefill_len",
        "decode_len",
        "prefill_latency",
        "decode_latency",
        "tokens_per_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "memory_mb",
        "cpu_pct",
        "temp_c",
        "power_w",
        "timestamp",
        "source",
    ]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse profiling log and export results.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to profiling.log")
    parser.add_argument(
        "--output",
        type=str,
        default="profiling_results/log_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    log_text = log_path.read_text()
    records = parse_log(log_text)
    write_csv(records, Path(args.output))


if __name__ == "__main__":
    main()

