import os
import sys
import time
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Ensure core modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.gps_parser import parse_nmea
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
    """Centralized configuration for easy deployment adjustments"""
    # Alert settings
    ALERT_DISTANCE_MILES = 0.15
    POI_COOLDOWN_MINUTES = 30  # Prevent re-alerting same POI too soon
    
    # GPS settings - Auto-detect platform
    IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model')
    GPS_SERIAL_PORT = "/dev/ttyAMA0" if IS_RASPBERRY_PI else "/dev/ttys009"
    GPS_TIMEOUT_SECONDS = 10  # Max time to wait for valid GPS data
    
    # File paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    POI_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'poi.json')
    
    # Error handling
    MAX_CONSECUTIVE_ERRORS = 5
    RETRY_DELAY_SECONDS = 2


def clean_llm_output(raw_output: str) -> str:
    """
    Removes <think> blocks, markdown formatting, and cleans text for TTS.
    
    Args:
        raw_output: Raw LLM response with potential think tags and markdown
        
    Returns:
        Clean text suitable for TTS output
    """
    if not raw_output:
        return ""
    
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown headers (###, ##, #)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove bold/italic markdown (**text**, *text*, __text__, _text_)
    cleaned = re.sub(r'\*\*([^\*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)
    cleaned = re.sub(r'\*([^\*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)
    
    # Remove bullet points and list markers
    cleaned = re.sub(r'^\s*[-*+]\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove code blocks
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    
    # Remove excessive newlines and whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Optional: Truncate very long responses for TTS (max ~500 words)
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
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logging.info("[SYSTEM] Initializing AI Tour Guide 🗺️")
            logging.info(f"[PLATFORM] Running on {'Raspberry Pi' if self.config.IS_RASPBERRY_PI else 'macOS/Development'}")
            logging.info(f"[GPS] Using port: {self.config.GPS_SERIAL_PORT}")
            
            # Initialize TTS system
            self.tts = HybridTTS()
            logging.info("[INIT] TTS system initialized ✅")
            
            # Initialize POI handler
            self.poi_handler = POIQuery(poi_file_path=self.config.POI_FILE_PATH)
            logging.info(f"[INIT] Loaded POI database from {self.config.POI_FILE_PATH} ✅")
            
            return True
            
        except FileNotFoundError as e:
            logging.error(f"[FATAL] Could not find the POI file at {self.config.POI_FILE_PATH}")
            logging.error(f"[FATAL] Error: {e}")
            return False
        except Exception as e:
            logging.error(f"[FATAL] Initialization failed: {e}")
            logging.exception(e)
            return False
    
    def should_alert_poi(self, poi_name: str) -> bool:
        """
        Check if enough time has passed since last alert for this POI.
        
        Args:
            poi_name: Name of the POI to check
            
        Returns:
            True if should alert, False if in cooldown period
        """
        if poi_name not in self.last_alert_times:
            return True
        
        time_since_last = datetime.now() - self.last_alert_times[poi_name]
        cooldown = timedelta(minutes=self.config.POI_COOLDOWN_MINUTES)
        
        if time_since_last < cooldown:
            remaining = cooldown - time_since_last
            logging.debug(f"[COOLDOWN] {poi_name} in cooldown for {remaining.seconds // 60} more minutes")
            return False
        
        return True
    
    def handle_poi_alert(self, nearest_poi: Dict[str, Any], distance: float):
        """
        Process and announce a POI alert.
        
        Args:
            nearest_poi: POI data dictionary
            distance: Distance to POI in miles
        """
        poi_name = nearest_poi.get("name", "Unnamed POI")
        
        try:
            logging.info(f"[ALERT] Approaching {poi_name} ({nearest_poi.get('city', 'N/A')}) - {distance:.2f} miles away")
            print("=" * 60)
            print(f"🎯 POINT OF INTEREST DETECTED: {poi_name}")
            print("=" * 60)
            
            # Generate LLM response
            raw_output = llm_inference.generate_response(nearest_poi)
            
            # Clean output for TTS
            clean_output = clean_llm_output(raw_output)
            
            # Display to console
            print("\n📢 AI Tour Guide says:")
            print("-" * 60)
            print(clean_output)
            print("-" * 60)
            print()
            
            # Speak the cleaned output
            if clean_output:
                self.tts.say(clean_output)
            else:
                logging.warning("[TTS] No clean output to speak")
            
            # Update tracking
            self.last_alerted_poi_name = poi_name
            self.last_alert_times[poi_name] = datetime.now()
            self.consecutive_errors = 0  # Reset error counter on success
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to process POI alert: {e}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.config.MAX_CONSECUTIVE_ERRORS:
                logging.critical(f"[FATAL] Too many consecutive errors ({self.consecutive_errors}). Check LLM/TTS configuration.")
                raise
    
    def process_gps_data(self, gps_data: Dict[str, Any]):
        """
        Process incoming GPS data and check for nearby POIs.
        
        Args:
            gps_data: GPS data dictionary with lat/lon
        """
        if "lat" not in gps_data or "lon" not in gps_data:
            logging.warning("[GPS] No valid coordinates received")
            return
        
        # Update GPS health tracking
        self.gps_last_valid_time = time.time()
        
        current_lat, current_lon = gps_data["lat"], gps_data["lon"]
        logging.debug(f"[GPS] Position: {current_lat:.6f}, {current_lon:.6f}")
        
        # Find nearest POI
        result = self.poi_handler.find_nearest(current_lat, current_lon)
        
        if not result:
            logging.debug("[POI] No POIs found in database")
            return
        
        nearest_poi, distance = result
        poi_name = nearest_poi.get("name", "Unnamed POI")
        
        # Check if within alert range
        if distance <= self.config.ALERT_DISTANCE_MILES:
            # Check if this is a new POI or cooldown has expired
            if poi_name != self.last_alerted_poi_name or self.should_alert_poi(poi_name):
                self.handle_poi_alert(nearest_poi, distance)
        else:
            # Reset alert state when moving away
            if poi_name == self.last_alerted_poi_name:
                logging.info(f"[INFO] Moved out of range for '{poi_name}'. Resetting alert state.")
                self.last_alerted_poi_name = None
    
    def check_gps_health(self):
        """Monitor GPS connection health"""
        if self.gps_last_valid_time:
            time_since_valid = time.time() - self.gps_last_valid_time
            if time_since_valid > self.config.GPS_TIMEOUT_SECONDS:
                logging.warning(f"[GPS] No valid data for {time_since_valid:.1f} seconds")
    
    def run(self):
        """Main application loop"""
        logging.info("[READY] Listening to GPS stream... 📡")
        logging.info(f"[CONFIG] Alert distance: {self.config.ALERT_DISTANCE_MILES} miles")
        logging.info(f"[CONFIG] POI cooldown: {self.config.POI_COOLDOWN_MINUTES} minutes")
        
        try:
            for gps_data in parse_nmea(port=self.config.GPS_SERIAL_PORT):
                self.process_gps_data(gps_data)
                self.check_gps_health()
                
        except KeyboardInterrupt:
            logging.info("\n[EXIT] Program terminated by user 👋")
        except Exception as e:
            logging.exception(f"[ERROR] Unexpected error in main loop: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all systems"""
        if self.tts:
            try:
                self.tts.stop()
                logging.info("[CLOSE] TTS engine shut down cleanly ✅")
            except Exception as e:
                logging.error(f"[ERROR] Error during TTS shutdown: {e}")


def main():
    """
    Entry point for the AI Tour Guide application.
    Runs the main proximity alert loop.
    """
    config = Config()
    system = TourGuideSystem(config)
    
    if not system.initialize():
        logging.error("[FATAL] System initialization failed. Exiting.")
        sys.exit(1)
    
    system.run()


if __name__ == "__main__":
    main()