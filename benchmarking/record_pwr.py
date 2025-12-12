#!/home/pi/morph_env/.venv/bin/python
"""
Callback for rd-usb: receives power meter JSON and logs to CSV.

- Reads rd-usb payload JSON file (list of measurements).
- Appends timestamp, power (W) rows into rd_usb_samples.csv in the lab/benchmark results dir.
- Moves raw JSON payloads into /tmp/rd_usb_payloads for debugging (optional).
"""

import json
import sys
import csv
import os
import time
import shutil
from pathlib import Path

# Where to keep the aggregated CSV
RESULTS_DIR = Path("/home/pi/AI-TOUR-GUIDE/benchmarking/results")
CSV_PATH = RESULTS_DIR / "rd_usb_samples.csv"

# Optional: archive JSON payloads under /tmp
PAYLOAD_ARCHIVE_DIR = Path("/tmp/rd_usb_payloads")
MAX_PAYLOADS = 200


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def append_samples(json_file: Path):
    # Load list of measurements from rd-usb
    try:
        with json_file.open() as f:
            measurements = json.load(f)
    except Exception as e:
        print(f"[record_pwr] Error reading {json_file}: {e}", file=sys.stderr)
        return

    if not isinstance(measurements, list):
        print(f"[record_pwr] Unexpected payload type in {json_file}, expected list.", file=sys.stderr)
        return

    # Open CSV and create header if missing
    csv_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["timestamp", "power_w"])

        for m in measurements:
            try:
                # Some firmwares send "timestamp", others may not; fall back to current time
                ts = m.get("timestamp", time.time())

                # Power in watts; adjust keys if your TC66 payload uses different names
                # Common rd-usb TC66 fields: "power", "voltage", "current"
                if "power" in m and m["power"] is not None:
                    power_w = float(m["power"])
                else:
                    # voltage (V) * current (A) if those are present
                    v = float(m.get("voltage", 0.0))
                    i = float(m.get("current", 0.0))
                    power_w = v * i

                writer.writerow([ts, power_w])
            except Exception as e:
                print(f"[record_pwr] Error parsing measurement in {json_file}: {e}", file=sys.stderr)
                continue


def archive_payload(json_file: Path):
    """
    Move the original rd-usb JSON payload into /tmp/rd_usb_payloads
    and keep only the newest MAX_PAYLOADS files (by mtime).
    """
    try:
        PAYLOAD_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        # Move file into archive dir
        dest = PAYLOAD_ARCHIVE_DIR / json_file.name
        try:
            shutil.move(str(json_file), str(dest))
        except FileNotFoundError:
            # If rd-usb already removed it or it never existed, just ignore
            return

        # Enforce retention: keep only newest MAX_PAYLOADS files
        files = [p for p in PAYLOAD_ARCHIVE_DIR.iterdir() if p.is_file()]
        if len(files) <= MAX_PAYLOADS:
            return

        # Sort by modification time ascending; delete oldest beyond cap
        files.sort(key=lambda p: p.stat().st_mtime)
        to_delete = files[0 : len(files) - MAX_PAYLOADS]
        for old in to_delete:
            try:
                old.unlink()
            except OSError as e:
                print(f"[record_pwr] Failed to delete old payload {old}: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[record_pwr] archive_payload error: {e}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage: record_pwr.py <json_file>", file=sys.stderr)
        sys.exit(1)

    json_path = Path(sys.argv[1])

    ensure_dirs()
    append_samples(json_path)
    archive_payload(json_path)


if __name__ == "__main__":
    main()

