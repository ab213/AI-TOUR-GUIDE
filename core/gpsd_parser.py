# core/gpsd_parser.py
"""
gpsd_parser.py — reads live GPS data from gpsd and yields structured dicts
compatible with the NMEA parser (lat/lon/time/speed/alt).
"""

from typing import Generator, Dict, Optional
import gps
import time
import logging

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

            # gpsd returns many report classes — we only want TPV (Time-Position-Velocity)
            if report["class"] != "TPV":
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
