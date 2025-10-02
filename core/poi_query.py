# poi_query.py

import os
import json
import math
from typing import Dict, Any, Tuple, List, Optional
import numba

@numba.njit(fastmath=True, cache=True)
def haversine(lat1: float, lon1: float, lat2: float, lon2: float, miles: bool = False) -> float:
    """
    Calculates the great-circle distance between two points on Earth.

    This high-performance version is compiled with Numba for speed, making it
    suitable for real-time calculations.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.
        miles (bool): If True, returns the distance in miles. Otherwise, in kilometers.

    Returns:
        float: The distance between the two points.
    """
    DEG_TO_RAD = math.pi / 180.0
    lat1_rad, lon1_rad = lat1 * DEG_TO_RAD, lon1 * DEG_TO_RAD
    lat2_rad, lon2_rad = lat2 * DEG_TO_RAD, lon2 * DEG_TO_RAD

    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad

    h = math.sin(dlat * 0.5)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon * 0.5)**2
    distance = 2.0 * math.asin(math.sqrt(h))

    return (3958.7613 if miles else 6371.0088) * distance


class POIQuery:
    """
    Manages loading, spatially indexing, and querying Points of Interest (POIs).

    Upon initialization, this class loads POIs from a JSON file and builds a
    spatial grid index for efficient nearest-neighbor searches.
    """
    def __init__(self, poi_file_path: str, precision: float = 0.01):
        """
        Initializes the POI handler.

        Args:
            poi_file_path (str): The absolute or relative path to the POI JSON file.
            precision (float): The granularity of the spatial grid. Smaller values
                               create a finer grid but may use more memory.
        """
        self.precision = precision
        print("Loading and indexing Points of Interest...")
        pois_data = self._load_pois(poi_file_path)
        self.grid = self._index_pois(pois_data)
        print(f"{len(pois_data)} POIs successfully indexed.")

    def _load_pois(self, path: str) -> Dict[str, Any]:
        """Loads POIs from the specified JSON file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"POI file not found at path: {path}")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _grid_hash(lat: float, lon: float, precision: float) -> Tuple[int, int]:
        """Maps a coordinate to a discrete grid cell."""
        return (int(lat / precision), int(lon / precision))

    def _index_pois(self, pois: Dict[str, Any]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
        """Builds the spatial index from a dictionary of POIs."""
        grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for coord_str, info in pois.items():
            try:
                # Safely evaluate string representation of a list/tuple like "[-84.1, 33.7]"
                lon, lat = json.loads(coord_str)
            except (json.JSONDecodeError, ValueError, TypeError):
                print(f"Warning: Skipping malformed POI coordinate key: {coord_str}")
                continue
            
            info['lat'] = float(lat)
            info['lon'] = float(lon)
            cell = self._grid_hash(info['lat'], info['lon'], self.precision)
            grid.setdefault(cell, []).append(info)
        return grid

    def find_nearest(self, lat: float, lon: float) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Finds the nearest POI to the given coordinates using the spatial index.

        It searches the coordinate's grid cell and all 8 adjacent cells to ensure
        the closest POI is found, even if it's across a cell boundary.

        Args:
            lat (float): The current latitude.
            lon (float): The current longitude.

        Returns:
            Optional[Tuple[Dict[str, Any], float]]: A tuple containing the nearest
            POI's data and its distance in miles, or None if no POIs are nearby.
        """
        current_cell = self._grid_hash(lat, lon, self.precision)
        candidates = []
        for d_lat in [-1, 0, 1]:
            for d_lon in [-1, 0, 1]:
                adj_cell = (current_cell[0] + d_lat, current_cell[1] + d_lon)
                candidates.extend(self.grid.get(adj_cell, []))

        if not candidates:
            return None

        min_dist = float('inf')
        nearest_poi = None
        for poi in candidates:
            dist_miles = haversine(lat, lon, poi['lat'], poi['lon'], miles=True)
            if dist_miles < min_dist:
                min_dist = dist_miles
                nearest_poi = poi

        return nearest_poi, min_dist