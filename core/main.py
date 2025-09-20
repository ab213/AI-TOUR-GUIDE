from gps_parser import parse_nmea, haversine

# Static reference point: Lenox Mall (Atlanta)
LENOX_LAT = 33.8467
LENOX_LON = -84.3637

if __name__ == "__main__":
    print("Starting real-time GPS distance calculation...")

    for data in parse_nmea():  # blocks until GPS data arrives
        if "lat" in data and "lon" in data:
            # Distance in kilometers (set miles=True for miles)
            dist_km = haversine(
                LENOX_LAT, LENOX_LON,
                data["lat"], data["lon"],
                miles=False
            )

            print(
                f"Time: {data.get('time')}, "
                f"Lat: {data['lat']:.6f}, Lon: {data['lon']:.6f}, "
                f"Distance from Lenox Mall: {dist_km:.3f} km"
            )
