import os
import sys
import time
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Ensure core modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# === Dynamic GPS parser import ===
try:
    from core.gpsd_parser import parse_gpsd as parse_gps
    GPS_MODE = "gpsd"
except ImportError:
    from core.gpsd_parser import parse_nmea as parse_gps
    GPS_MODE = "nmea"

from core.poi_query import POIQuery
from llm import llm_inference
from audio.tts import HybridTTS

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# === Configuration ===
class Config:
    """Centralized configuration for deployment adjustments"""
    ALERT_DISTANCE_MILES = 0.25
    POI_COOLDOWN_MINUTES = 30
    IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model')
    GPS_SERIAL_PORT = "/dev/ttyAMA0" if IS_RASPBERRY_PI else "/dev/ttys009"
    GPS_TIMEOUT_SECONDS = 10
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    POI_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'poi.json')
    MAX_CONSECUTIVE_ERRORS = 5
    RETRY_DELAY_SECONDS = 2


def clean_llm_output(raw_output: str) -> str:
    """Removes <think> blocks, markdown formatting, and cleans text for TTS."""
    if not raw_output:
        return ""

    cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(\*\*|__|\*|_)(.*?)\1', r'\2', cleaned)
    cleaned = re.sub(r'^\s*[-*+]\s+|\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()

    words = cleaned.split()
    if len(words) > 500:
        cleaned = ' '.join(words[:500]) + "..."
        logging.warning(f"[TTS] Response truncated from {len(words)} to 500 words")

    return cleaned


class TourGuideSystem:
    """Main system class for managing the tour guide application"""

    def __init__(self, config: Config):
        self.config = config
        self.tts = None
        self.poi_handler = None
        self.last_alerted_poi_name: Optional[str] = None
        self.last_alert_times: Dict[str, datetime] = {}
        self.consecutive_errors = 0
        self.gps_last_valid_time = None

    def initialize(self) -> bool:
        """Initialize system components"""
        try:
            logging.info("[SYSTEM] Initializing AI Tour Guide 🗺️")
            logging.info(f"[PLATFORM] {'Raspberry Pi' if self.config.IS_RASPBERRY_PI else 'macOS/Dev'}")
            logging.info(f"[GPS] Mode: {GPS_MODE.upper()}")

            self.tts = HybridTTS()
            logging.info("[INIT] TTS initialized ✅")

            self.poi_handler = POIQuery(poi_file_path=self.config.POI_FILE_PATH)
            logging.info(f"[INIT] Loaded POI database ✅")

            return True
        except FileNotFoundError as e:
            logging.error(f"[FATAL] Missing POI file at {self.config.POI_FILE_PATH}")
            logging.error(e)
            return False
        except Exception as e:
            logging.error(f"[FATAL] Initialization failed: {e}")
            logging.exception(e)
            return False

    def should_alert_poi(self, poi_name: str) -> bool:
        """Return True if cooldown expired for a POI"""
        if poi_name not in self.last_alert_times:
            return True

        elapsed = datetime.now() - self.last_alert_times[poi_name]
        cooldown = timedelta(minutes=self.config.POI_COOLDOWN_MINUTES)
        return elapsed >= cooldown

    def handle_poi_alert(self, nearest_poi: Dict[str, Any], distance: float):
        """Announce a POI using LLM + TTS"""
        poi_name = nearest_poi.get("name", "Unnamed POI")
        try:
            logging.info(f"[ALERT] Near {poi_name} ({distance:.2f} mi)")
            print("=" * 60)
            print(f"🎯 POINT OF INTEREST DETECTED: {poi_name}")
            print("=" * 60)

            raw_output = llm_inference.generate_response(nearest_poi)
            clean_output = clean_llm_output(raw_output)

            print("\n📢 AI Tour Guide says:")
            print("-" * 60)
            print(clean_output)
            print("-" * 60)

            if clean_output:
                self.tts.say(clean_output)
                self.tts.queue.join()

            self.last_alerted_poi_name = poi_name
            self.last_alert_times[poi_name] = datetime.now()
            self.consecutive_errors = 0

        except Exception as e:
            self.consecutive_errors += 1
            logging.error(f"[ERROR] POI alert failed: {e}")
            if self.consecutive_errors >= self.config.MAX_CONSECUTIVE_ERRORS:
                logging.critical("[FATAL] Too many consecutive errors.")
                raise

    def process_gps_data(self, gps_data: Dict[str, Any]):
        """Process a GPS reading"""
        if not gps_data.get("lat") or not gps_data.get("lon"):
            logging.warning("[GPS] Invalid GPS data")
            return

        self.gps_last_valid_time = time.time()
        current_lat, current_lon = gps_data["lat"], gps_data["lon"]
        logging.debug(f"[GPS] Position: {current_lat:.6f}, {current_lon:.6f}")

        result = self.poi_handler.find_nearest(current_lat, current_lon)
        if not result:
            return

        nearest_poi, distance = result
        poi_name = nearest_poi.get("name", "Unknown POI")

        if distance <= self.config.ALERT_DISTANCE_MILES:
            if poi_name != self.last_alerted_poi_name or self.should_alert_poi(poi_name):
                self.handle_poi_alert(nearest_poi, distance)
        elif poi_name == self.last_alerted_poi_name:
            logging.info(f"[INFO] Moved out of range for '{poi_name}' — resetting alert.")
            self.last_alerted_poi_name = None

    def check_gps_health(self):
        """Check if GPS stopped reporting"""
        if self.gps_last_valid_time:
            elapsed = time.time() - self.gps_last_valid_time
            if elapsed > self.config.GPS_TIMEOUT_SECONDS:
                logging.warning(f"[GPS] No valid data for {elapsed:.1f} seconds")

    def run(self):
        """Main runtime loop"""
        logging.info("[READY] Listening to GPS stream... 📡")
        logging.info(f"[CONFIG] Alert distance: {self.config.ALERT_DISTANCE_MILES} mi")
        logging.info(f"[CONFIG] POI cooldown: {self.config.POI_COOLDOWN_MINUTES} min")

        try:
            # gpsd automatically handles connection to hardware
            for gps_data in parse_gps(port=self.config.GPS_SERIAL_PORT) if GPS_MODE == "nmea" else parse_gps():
                self.process_gps_data(gps_data)
                self.check_gps_health()

        except KeyboardInterrupt:
            logging.info("\n[EXIT] User terminated the program 👋")
        except Exception as e:
            logging.exception(f"[ERROR] Unexpected main loop failure: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown"""
        if self.tts:
            try:
                self.tts.stop()
                logging.info("[CLOSE] TTS engine stopped ✅")
            except Exception as e:
                logging.error(f"[ERROR] During TTS shutdown: {e}")


def main():
    """Entry point"""
    config = Config()
    system = TourGuideSystem(config)

    if not system.initialize():
        logging.error("[FATAL] Initialization failed — exiting.")
        sys.exit(1)

    system.run()


if __name__ == "__main__":
    main()
