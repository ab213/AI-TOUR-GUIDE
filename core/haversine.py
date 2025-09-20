import math

def hav(theta):
    return (1-math.cos(theta)) / 2

def hav_outer(lat1, lat2, long1, long2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)
    return hav(lat2 - lat1) + math.cos(lat1)*math.cos(lat2)*hav(long2-long1)

def haversine(r, lat1, lat2, long1, long2):
    return 2 * r * math.asin(math.sqrt(hav_outer(lat1, lat2, long1, long2)))

if __name__ == "__main__":
    parisLat = 48.8575
    parisLong = 2.3514
    atlantaLat = 33.7501
    atlantaLong = -84.3885
    earth_rad_km = 6378.1
    distance_ex = haversine(earth_rad_km, parisLat, atlantaLat, parisLong, atlantaLong)
    print(distance_ex)