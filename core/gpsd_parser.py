"""
gpsd_parser.py — reads live GPS data from gpsd and yields structured dicts
compatible with the NMEA parser (lat/lon/time/speed/alt).
"""

from typing import Generator, Dict, Optional
import time
import logging
import json

# --- 🔧 Monkeypatch for gpsd JSONDecoder encoding bug ---
if "encoding" not in json.JSONDecoder.__init__.__code__.co_varnames:
    old_init = json.JSONDecoder.__init__

    def new_init(self, *args, **kwargs):
        kwargs.pop("encoding", None)  # drop deprecated arg safely
        old_init(self, *args, **kwargs)

    json.JSONDecoder.__init__ = new_init
# --------------------------------------------------------

import gps  # ⬅️ must come AFTER the patch

def parse_gpsd() -> Generator[Dict[str, Optional[float]], None, None]:
    """
    Continuously read GPS data from gpsd and yield structured dictionaries.
    Compatible with parse_nmea() format.

    Yields:
        dict: {
            "lat": float | None,
            "lon": float | None,
            "alt": float | None,
            "speed": float | None,   # m/s
            "climb": float | None,   # m/s
            "time": str | None
        }
    """
    session = gps.gps(mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
    logging.info("[GPSD] Connected to gpsd daemon ✅")

    while True:
        try:
            report = session.next()
            logging.debug(f"[GPSD] Report received: {report}")

            # gpsd returns many report classes — we only want TPV (Time-Position-Velocity)
            if report["class"] != "TPV":
                logging.debug(f"[GPSD] Skipped report class: {report.get('class')}")
                continue

            data = {
                "lat": getattr(report, "lat", None),
                "lon": getattr(report, "lon", None),
                "alt": getattr(report, "alt", None),
                "speed": getattr(report, "speed", None),
                "climb": getattr(report, "climb", None),
                "time": getattr(report, "time", None),
            }

            # Only yield if we got valid coordinates
            if data["lat"] is not None and data["lon"] is not None:
                yield data
            else:
                time.sleep(0.5)

        except StopIteration:
            logging.warning("[GPSD] Stream ended — reconnecting...")
            session = gps.gps(mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
            time.sleep(1)
        except KeyError:
            continue
        except Exception as e:
            logging.error(f"[GPSD] Error reading gpsd data: {e}")
            time.sleep(1)
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Starting GPSD parser test - press Ctrl+C to stop")

    try:
        count = 0
        for gps_data in parse_gpsd():
            print(f"GPS Data: {gps_data}")
            count += 1
            if count >= 10:  # Stop after 10 updates for test
                break
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in test: {e}")

    print("GPSD parser test completed")

