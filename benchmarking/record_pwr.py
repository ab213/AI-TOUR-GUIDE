"""
Callback for rd-usb: receives power meter JSON and logs to CSV.
Compatible with TC66C, UM25C, UM24C, UM34C (adjust parsing as needed).
"""

import json
import sys
import csv
import os

LOG_PATH = "$HOME/AI-TOUR-GUIDE/benchmarking/rd_usb_samples.csv"


def main():
    if len(sys.argv) < 2:
        print("Usage: on_receive.py <json_file>", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    
    try:
        with open(json_file) as f:
            measurements = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Create CSV header if needed
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "power"])
        
        # Parse measurements (format varies by meter, adjust field names as needed)
        for m in measurements:
            # TC66C/UM25C format: timestamp (Unix), voltage, current, power, ...
            try:
                timestamp = m.get("timestamp") or time.time()
                # Power in watts (may be under different keys depending on meter)
                power = m.get("power") or (m.get("voltage", 0) * m.get("current", 0) / 1000)
                writer.writerow([timestamp, power])
            except Exception as e:
                print(f"Error parsing measurement: {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()

