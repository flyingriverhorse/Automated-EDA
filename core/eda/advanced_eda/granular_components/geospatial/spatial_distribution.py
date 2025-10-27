"""Spatial distribution analysis component."""

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


class SpatialDistributionAnalysis:
    """Metadata and notebook code for spatial distribution visualisation."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Spatial Distribution Analysis",
            "description": "Plot geographic scatter and density heatmaps using latitude/longitude columns.",
            "category": "Geospatial Analysis",
            "complexity": "intermediate",
            "estimated_runtime": "3-6 seconds",
            "tags": ["geospatial", "visualisation", "density", "map"],
            "icon": "🗺️",
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

        required = [latitude, longitude]

        return f"""
# SPATIAL DISTRIBUTION ANALYSIS
import matplotlib.pyplot as plt

required = {required!r}
missing = [col for col in required if col not in df.columns]
if missing:
	raise ValueError(f"Missing required coordinate columns: {{missing}}")

geo_df = df[{required}].dropna()
geo_df = geo_df.apply(pd.to_numeric, errors='coerce').dropna()

if geo_df.empty:
	raise ValueError("No usable latitude/longitude pairs after coercion.")

sample = geo_df.sample(n=min(len(geo_df), 500), random_state=42)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample[{longitude!r}], sample[{latitude!r}], s=12, alpha=0.65, color="#2D9CDB")
ax.set_title("Geospatial Scatter (sampled)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.2)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
hb = ax.hexbin(sample[{longitude!r}], sample[{latitude!r}], gridsize=35, cmap='viridis', mincnt=1)
ax.set_title("Spatial Density (hexbin)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cb = fig.colorbar(hb, ax=ax)
cb.set_label("Point count")
plt.show()
"""


def get_component():
    return SpatialDistributionAnalysis
