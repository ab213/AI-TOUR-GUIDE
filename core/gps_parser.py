from serial import Serial
from pynmeagps import NMEAReader
import math
import numba

@numba.njit(fastmath=True, cache=True)
def haversine(lat1, lon1, lat2, lon2, miles=False):
    """
    Ultra-fast haversine distance using Numba JIT.
    Suitable for real-time GPS parsing.
    
    Args:
        lat1, lon1, lat2, lon2 : float (decimal degrees)
        miles : bool, if True → return miles, else kilometers.
    
    Returns:
        float: distance in chosen unit
    """
    # Convert degrees → radians
    DEG_TO_RAD = math.pi / 180.0
    lat1 *= DEG_TO_RAD
    lon1 *= DEG_TO_RAD
    lat2 *= DEG_TO_RAD
    lon2 *= DEG_TO_RAD

    # Deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    h = math.sin(dlat * 0.5)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon * 0.5)**2
    distance = 2.0 * math.asin(math.sqrt(h))

    # Scale by Earth radius
    return (3958.7613 if miles else 6371.0088) * distance

def parse_nmea(port="/dev/ttys011", baud=9600):
    """
    Continuously parses live NMEA data from a virtual or real GPS
    and yields structured data as dictionaries.
    
    Args:
        port (str): Serial port to connect to.
        baud (int): Baud rate for the GPS.
    
    Yields:
        dict: Parsed NMEA sentence data.
    """
    with Serial(port, baud, timeout=3) as stream:
        nmr = NMEAReader(stream)
        while True:
            try:
                raw_data, parsed_data = nmr.read()
                if parsed_data:
                    sentence = {"raw": raw_data.strip(), "type": parsed_data.msgID}

                    if parsed_data.msgID == "GGA":
                        sentence.update({
                            "time": parsed_data.time,
                            "lat": parsed_data.lat,
                            "lon": parsed_data.lon,
                            "alt": parsed_data.alt,
                            "sats": parsed_data.numSV
                        })
                    
                    elif parsed_data.msgID == "RMC":
                        sentence.update({
                            "time": parsed_data.time,
                            "status": parsed_data.status,
                            "lat": parsed_data.lat,
                            "lon": parsed_data.lon,
                            "speed_knots": parsed_data.spd,
                            "date": parsed_data.date
                        })

                    elif parsed_data.msgID == "GSA":
                        sentence.update({
                            "mode": parsed_data.mode,
                            "fix": parsed_data.navMode,
                            "pdop": parsed_data.pdop,
                            "hdop": parsed_data.hdop,
                            "vdop": parsed_data.vdop
                        })

                    elif parsed_data.msgID == "GSV":
                        sentence.update({
                            "num_msgs": parsed_data.numMsg,
                            "sats_in_view": parsed_data.numSV
                        })

                    yield sentence  

            except KeyboardInterrupt:
                print("\nStopping GPS parser.")
                break
            except Exception as e:
                yield {"error": str(e)}
                continue



if __name__ == "__main__":
    for d in parse_nmea():
        print(d)
