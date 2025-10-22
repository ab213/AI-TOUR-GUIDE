import time

# Change this to the FIRST socat PTY (your GPS "device")
GPS_PORT = "/dev/ttys011"

# Starting point: Five Points MARTA Station, Atlanta
lat, lon = 33.753746, -84.391502  
speed_kmh = 10.0   # going north at 40 km/h
heading = 0.0      # due north
altitude = 300.0   # meters
satellites = 8     # visible satellites
hdop = 0.9         # horizontal dilution of precision

def nmea_checksum(sentence: str) -> str:
    cs = 0
    for c in sentence:
        cs ^= ord(c)
    return f"{cs:02X}"

def deg2nmea(deg: float, is_lat: bool = True):
    d = int(abs(deg))
    m = (abs(deg) - d) * 60
    if is_lat:
        return f"{d:02d}{m:07.4f}", "N" if deg >= 0 else "S"
    else:
        return f"{d:03d}{m:07.4f}", "E" if deg >= 0 else "W"

with open(GPS_PORT, "w") as gps:
    while True:
        # Move ~11m north per second
        lat += (speed_kmh * 1000 / 3600) / 111_111.0  

        # Current UTC time
        utc_time = time.strftime("%H%M%S", time.gmtime())
        date_str = time.strftime("%d%m%y", time.gmtime())

        # Convert to NMEA
        lat_str, ns = deg2nmea(lat, is_lat=True)
        lon_str, ew = deg2nmea(lon, is_lat=False)
        spd_knots = speed_kmh / 1.852

        # ---------------- GPRMC ----------------
        rmc_body = (
            f"GPRMC,{utc_time}.00,A,{lat_str},{ns},{lon_str},{ew},"
            f"{spd_knots:.1f},{heading:.1f},{date_str},,,A"
        )
        rmc_sentence = f"${rmc_body}*{nmea_checksum(rmc_body)}"

        # ---------------- GPGGA ----------------
        gga_body = (
            f"GPGGA,{utc_time}.00,{lat_str},{ns},{lon_str},{ew},1,"
            f"{satellites:02d},{hdop:.1f},{altitude:.1f},M,0.0,M,,"
        )
        gga_sentence = f"${gga_body}*{nmea_checksum(gga_body)}"

        # Write to virtual port
        gps.write(rmc_sentence + "\r\n")
        gps.write(gga_sentence + "\r\n")
        gps.flush()

        print("Sent:", rmc_sentence)
        print("Sent:", gga_sentence)

        time.sleep(1)
