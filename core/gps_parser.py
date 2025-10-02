# gps_parser.py

from typing import Dict, Generator
from serial import Serial, SerialException
from pynmeagps import NMEAReader

def parse_nmea(port: str = "/dev/ttys031", baud: int = 9600) -> Generator[Dict, None, None]:
    """
    Continuously connects to and parses live NMEA data from a GPS device.

    This generator function opens a serial connection and reads incoming data line by
    line, parsing valid NMEA sentences into a structured dictionary format. It handles
    various common sentence types like GGA, RMC, GSA, and GSV.

    Args:
        port (str): The serial port to connect to (e.g., '/dev/ttyUSB0' on Linux
                    or 'COM3' on Windows).
        baud (int): The baud rate for the serial communication.

    Yields:
        Generator[Dict, None, None]: A dictionary representing the parsed data from
                                     a single NMEA sentence.
    """
    print(f"Attempting to connect to GPS on port {port} at {baud} baud...")
    try:
        stream = Serial(port, baud, timeout=3)
    except SerialException as e:
        print(f"Error: Could not open serial port '{port}'.")
        print(f"Details: {e}")
        print("Please ensure the port is correct and not in use by another program.")
        return

    with stream:
        print("Successfully connected to GPS. Reading data...")
        nmr = NMEAReader(stream)

        while True:
            try:
                (raw_data, parsed_data) = nmr.read()
                if not parsed_data:
                    continue

                sentence = {"raw": raw_data.strip().decode('utf-8'), "type": parsed_data.msgID}

                # Add specific fields based on message type
                if parsed_data.msgID == "GGA":
                    sentence.update({
                        "time": parsed_data.time,
                        "lat": parsed_data.lat,
                        "lon": parsed_data.lon,
                        "alt": parsed_data.alt,
                        "quality": parsed_data.quality,
                        "sats": parsed_data.numSV,
                    })
                elif parsed_data.msgID == "RMC":
                    sentence.update({
                        "time": parsed_data.time,
                        "status": parsed_data.status,
                        "lat": parsed_data.lat,
                        "lon": parsed_data.lon,
                        "speed_knots": parsed_data.spd,
                        "date": parsed_data.date,
                    })
                
                # Only yield sentences that contain location data for our use case
                if "lat" in sentence and "lon" in sentence:
                    yield sentence

            except (IOError, ValueError, TypeError) as e:
                print(f"An error occurred during parsing: {e}")
                continue


if __name__ == "__main__":
    print("--- Testing gps_parser.py module independently ---")
    print("This will print the first 5 parsed sentences with location data.")
    print("To stop, press Ctrl+C.")
    
    try:
        count = 0
        for sentence_data in parse_nmea():
            if sentence_data and sentence_data.get("lat"):
                print(sentence_data)
                count += 1
                if count >= 5:
                    break
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    print("\n--- GPS parser test complete ---")