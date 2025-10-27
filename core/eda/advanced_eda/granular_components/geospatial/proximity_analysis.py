"""Geospatial proximity analysis component."""

from typing import Any, Dict, Iterable, Optional


LATITUDE_HINTS = ("latitude", "lat", "lat_deg", "latitud")
LONGITUDE_HINTS = ("longitude", "lon", "lng", "long", "lon_deg")


def _detect_column(columns: Iterable[str], hints: Iterable[str]) -> Optional[str]:
    for hint in hints:
        for column in columns:
            col_lower = column.lower()
            if col_lower == hint or hint in col_lower:
                return column
    return None


class GeospatialProximityAnalysis:
    """Metadata and notebook code for proximity analysis."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Geospatial Proximity Analysis",
            "description": "Measure distances to a reference point and highlight coverage within a custom radius.",
            "category": "Geospatial Analysis",
            "complexity": "intermediate",
            "estimated_runtime": "4-8 seconds",
            "tags": ["geospatial", "distance", "proximity", "radius"],
            "icon": "📍",
        }

    def validate_data_compatibility(self, data_preview: Dict[str, Any] | None = None) -> bool:
        if not data_preview:
            return True
        columns = data_preview.get("columns") or []
        latitude_column = _detect_column(columns, LATITUDE_HINTS)
        longitude_column = _detect_column(columns, LONGITUDE_HINTS)
        return bool(latitude_column and longitude_column)

    def generate_code(self, data_preview: Dict[str, Any] | None = None) -> str:
        latitude = "latitude"
        longitude = "longitude"

        if data_preview:
            columns = data_preview.get("columns", [])
            latitude_match = _detect_column(columns, LATITUDE_HINTS)
            longitude_match = _detect_column(columns, LONGITUDE_HINTS)
            if latitude_match:
                latitude = latitude_match
            if longitude_match:
                longitude = longitude_match

        template = """
# GEOSPATIAL PROXIMITY ANALYSIS
import numpy as np
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = [__LATITUDE_COL__, __LONGITUDE_COL__]
missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Missing required coordinate columns: {missing}")

latitudes = pd.to_numeric(df[__LATITUDE_COL__], errors='coerce')
longitudes = pd.to_numeric(df[__LONGITUDE_COL__], errors='coerce')

clean_df = pd.DataFrame({
    "latitude": latitudes,
    "longitude": longitudes,
}).dropna()

if clean_df.empty:
    raise ValueError("No valid coordinate pairs remain after cleaning.")

# Configure proximity options
PROXIMITY_RADIUS_KM = 5.0  # Adjust or override as needed
REFERENCE_LATITUDE = None  # Provide a manual latitude (decimal degrees)
REFERENCE_LONGITUDE = None  # Provide a manual longitude (decimal degrees)

if REFERENCE_LATITUDE is None or REFERENCE_LONGITUDE is None:
    REFERENCE_LATITUDE = clean_df["latitude"].mean()
    REFERENCE_LONGITUDE = clean_df["longitude"].mean()
    reference_source = "dataset centroid"
else:
    reference_source = "manual"

print(f"📍 Reference point: ({REFERENCE_LATITUDE:.4f}, {REFERENCE_LONGITUDE:.4f}) • Source: {reference_source}")

ref_lat_rad = np.radians(REFERENCE_LATITUDE)
ref_lon_rad = np.radians(REFERENCE_LONGITUDE)

lat_rad = np.radians(clean_df["latitude"].to_numpy())
lon_rad = np.radians(clean_df["longitude"].to_numpy())

delta_lat = lat_rad - ref_lat_rad
delta_lon = lon_rad - ref_lon_rad

a = np.sin(delta_lat / 2.0) ** 2 + np.cos(ref_lat_rad) * np.cos(lat_rad) * np.sin(delta_lon / 2.0) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))
distances_km = 6371.0 * c

clean_df["distance_km"] = distances_km

print("\n📊 Distance Summary")
print("  • Mean distance (km):", round(float(clean_df["distance_km"].mean()), 3))
print("  • Median distance (km):", round(float(clean_df["distance_km"].median()), 3))
print("  • 90th percentile (km):", round(float(clean_df["distance_km"].quantile(0.9)), 3))

if PROXIMITY_RADIUS_KM and PROXIMITY_RADIUS_KM > 0:
    within_radius = (clean_df["distance_km"] <= PROXIMITY_RADIUS_KM)
    count_within = int(within_radius.sum())
    share_within = (count_within / len(clean_df)) * 100
    print(f"  • Within {PROXIMITY_RADIUS_KM} km: {count_within} rows ({share_within:.1f}%)")
else:
    print("  • Configure PROXIMITY_RADIUS_KM to measure coverage around the reference point.")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(clean_df["distance_km"], bins=30, color="#0984E3", alpha=0.85)
ax.set_xlabel("Distance to reference (km)")
ax.set_ylabel("Count")
ax.set_title("Distribution of distances")

if PROXIMITY_RADIUS_KM and PROXIMITY_RADIUS_KM > 0:
    ax.axvline(PROXIMITY_RADIUS_KM, color="#E17055", linestyle='--', linewidth=2, label=f"Radius {PROXIMITY_RADIUS_KM} km")
    ax.legend()

plt.tight_layout()
plt.show()

print("\nTop 10 closest observations:")
display(clean_df.sort_values("distance_km").head(10))
"""

        return template.replace("__LATITUDE_COL__", repr(latitude)).replace("__LONGITUDE_COL__", repr(longitude))


def get_component():
    return GeospatialProximityAnalysis
