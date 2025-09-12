from serial import Serial
from pynmeagps import NMEAReader

def parse_nmea(port="/dev/ttys005", baud=9600):
    """
    Parses live NMEA data from a virtual or real GPS.
    """
    with Serial(port, baud, timeout=3) as stream:
        nmr = NMEAReader(stream)
        print(f"Listening on {port} at {baud} baud...\n")
        while True:
            try:
                raw_data, parsed_data = nmr.read()  # Read next sentence
                if parsed_data:
                    # Print raw NMEA
                    print(f"RAW: {raw_data.strip()}")

                    # Handle by sentence type
                    if parsed_data.msgID == "GGA":
                        print(f"[GGA] Time: {parsed_data.time} "
                              f"Lat: {parsed_data.lat} "
                              f"Lon: {parsed_data.lon} "
                              f"Alt: {parsed_data.alt}m "
                              f"Sats: {parsed_data.numSV}")
                    
                    elif parsed_data.msgID == "RMC":
                        print(f"[RMC] Time: {parsed_data.time} "
                            f"Status: {parsed_data.status} "
                            f"Lat: {parsed_data.lat} "
                            f"Lon: {parsed_data.lon} "
                            f"Speed: {parsed_data.spd}kn "
                            f"Date: {parsed_data.date}")

                    
                    elif parsed_data.msgID == "GSA":
                        print(f"[GSA] Mode: {parsed_data.mode} "
                              f"Fix: {parsed_data.navMode} "
                              f"PDOP: {parsed_data.pdop} "
                              f"HDOP: {parsed_data.hdop} "
                              f"VDOP: {parsed_data.vdop}")
                    
                    elif parsed_data.msgID == "GSV":
                        print(f"[GSV] Msgs: {parsed_data.numMsg} "
                              f"Sat in View: {parsed_data.numSV}")

                    print("-" * 60)

            except KeyboardInterrupt:
                print("\nStopping GPS parser.")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    parse_nmea()
