import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.gps_parser import parse_nmea
from core.poi_query import POIQuery
from llm import llm_inference
from audio.tts import init_tts, speak

def main():
    """
    Runs the main AI Tour Guide proximity alert application loop.

    This function initializes the POI query handler, listens to the GPS stream,
    and triggers alerts when the current location is within a specified
    distance of a known Point of Interest.
    """
    ## --- Configuration ---
    # The distance in miles to trigger an alert 
    ALERT_DISTANCE_MILES = 0.15

    # The serial port for your real or virtual GPS.
    # This should match the port used by your virtual_gps.py or physical device.
    GPS_SERIAL_PORT = "/dev/ttys031" 

    # Construct the absolute path to the POI file.
    # It navigates up one directory ('..') from 'core' to the project root, then into 'data'.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    POI_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'poi.json')

    ## --- Initialization ---
    try:
        # Create an instance of the POI handler, which loads and indexes the data.
        poi_handler = POIQuery(poi_file_path=POI_FILE_PATH)
    except FileNotFoundError as e:
        print(f" Fatal Error: Could not find the POI file.")
        print(f"   Checked path: {POI_FILE_PATH}")
        print(f"   Details: {e}")
        return

    # Initialize TTS system
    tts_state = init_tts()

    # State variable to track the last alerted POI to prevent alert spam
    last_alerted_poi_name = None

    ## --- Main Application Loop ---
    # This loop continuously processes incoming GPS data.
    for gps_data in parse_nmea(port=GPS_SERIAL_PORT):
        # We only proceed if we have a valid dictionary with latitude and longitude.
        if "lat" in gps_data and "lon" in gps_data:
            current_lat, current_lon = gps_data["lat"], gps_data["lon"]

            # Find the closest POI from our indexed grid.
            result = poi_handler.find_nearest(current_lat, current_lon)

            if result:
                nearest_poi, distance = result
                poi_name = nearest_poi.get("name", "Unnamed POI")

                # 1. Check if we are inside the alert radius.
                if distance <= ALERT_DISTANCE_MILES:
                    # 2. Trigger an alert only if it's a *new* POI we are close to.
                    if poi_name != last_alerted_poi_name:
                        print(f"Approaching: {poi_name} (in {nearest_poi.get('city', 'N/A')})")
                        print(f"Current Distance: {distance:.2f} miles")
                        print("--------------------------------\n")
                        output = llm_inference.generate_response(nearest_poi)
                        print("AI Tour Guide says:\n", output)
                        # Speak the response via TTS
                        speak(tts_state, output)
                        # Update state to prevent re-alerting for the same place.
                        last_alerted_poi_name = poi_name 
                else:
                    # 3. If we have moved away from the POI we last alerted for, reset its state.
                    #    This allows us to be alerted again if we re-enter its radius later.
                    if poi_name == last_alerted_poi_name:
                        print(f"INFO: Moved out of range for '{poi_name}'. Alert status reset.")
                        last_alerted_poi_name = None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")