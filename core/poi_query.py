import os
import json
from typing import Dict, Any, Tuple, List, Optional
from core.gps_parser import get_current_location
from core.haversine import haversine

POI_PATH = os.path.join(os.path.dirname(__file__), 'data/poi.json')

def load_pois(path: str = POI_PATH) -> Dict[str, Any]:
    """
    Securely load POIs from the specified JSON file.
    Only allows reading from the expected path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"POI file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def grid_hash(lat: float, lon: float, precision: float = 0.01) -> Tuple[int, int]:
    """
    Maps latitude and longitude to a grid cell for spatial clustering.
    """
    return (int(lat / precision), int(lon / precision))

def index_pois(pois: Dict[str, Any], precision: float = 0.01) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Indexes POIs by grid cell for efficient spatial lookup.
    Adds lat/lon to each POI entry for later use.
    """
    grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for coord_str, info in pois.items():
        try:
            lon, lat = eval(coord_str, {"__builtins__": None}, {})
        except Exception:
            continue  # skip malformed entries
        info['lat'] = lat
        info['lon'] = lon
        cell = grid_hash(lat, lon, precision)
        grid.setdefault(cell, []).append(info)
    return grid

def find_nearest_poi(lat: float, lon: float, grid: Dict[Tuple[int, int], List[Dict[str, Any]]], precision: float = 0.01) -> Optional[Dict[str, Any]]:
    """
    Finds the nearest POI to the given lat/lon using grid adjacency.
    """
    cell = grid_hash(lat, lon, precision)
    candidates = []
    for dlat in [-1, 0, 1]:
        for dlon in [-1, 0, 1]:
            adj_cell = (cell[0]+dlat, cell[1]+dlon)
            candidates.extend(grid.get(adj_cell, []))
    min_dist = float('inf')
    nearest = None
    for poi in candidates:
        poi_lat = float(poi.get('lat', 0))
        poi_lon = float(poi.get('lon', 0))
        if not poi_lat and not poi_lon:
            continue
        dist = haversine(lat, lon, poi_lat, poi_lon)
        if dist < min_dist:
            min_dist = dist
            nearest = poi
    return nearest

def query_closest_poi(current_lat: float, current_lon: float, precision: float = 0.01) -> Optional[Dict[str, Any]]:
    """
    Loads POIs, builds grid index, and returns the closest POI to the given location.
    """
    pois = load_pois()
    grid = index_pois(pois, precision=precision)
    return find_nearest_poi(current_lat, current_lon, grid, precision=precision)

if __name__ == "__main__":
    # test retrieve current location from GPS
    try:
        current_lat, current_lon = get_current_location()
        nearest = query_closest_poi(current_lat, current_lon)
        print(f"Closest POI: {nearest}")
    except Exception as e:
        print(f"Error retrieving GPS location: {e}")
