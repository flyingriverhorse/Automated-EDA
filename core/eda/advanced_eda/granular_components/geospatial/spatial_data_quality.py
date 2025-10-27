"""Spatial data quality analysis component."""

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


class SpatialDataQualityAnalysis:
    """Metadata and notebook code for auditing spatial data quality."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Spatial Data Quality Analysis",
            "description": "Identify missing, invalid, or duplicate coordinate pairs and visualise distributions.",
            "category": "Geospatial Analysis",
            "complexity": "intermediate",
            "estimated_runtime": "3-6 seconds",
            "tags": ["geospatial", "quality", "validation", "coordinates"],
            "icon": "🧭",
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
# SPATIAL DATA QUALITY ANALYSIS
import matplotlib.pyplot as plt

required = [__LATITUDE_COL__, __LONGITUDE_COL__]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required coordinate columns: {missing}")

lat_raw = df[__LATITUDE_COL__]
lon_raw = df[__LONGITUDE_COL__]

lat_numeric = pd.to_numeric(lat_raw, errors='coerce')
lon_numeric = pd.to_numeric(lon_raw, errors='coerce')

summary = {
    "rows": len(df),
    "valid_pairs": int((lat_numeric.notna() & lon_numeric.notna()).sum()),
    "missing_lat": int(lat_raw.isna().sum()),
    "missing_lon": int(lon_raw.isna().sum()),
    "non_numeric_lat": int(lat_raw.notna().sum() - lat_numeric.notna().sum()),
    "non_numeric_lon": int(lon_raw.notna().sum() - lon_numeric.notna().sum()),
    "out_of_range_lat": int((lat_numeric.abs() > 90).sum()),
    "out_of_range_lon": int((lon_numeric.abs() > 180).sum()),
}

print("🧭 Spatial Data Quality Summary")
for key, value in summary.items():
    print(f"  - {key.replace('_', ' ').title()}: {value}")

issues = []
if summary["valid_pairs"] == 0:
    issues.append("No valid coordinate pairs after cleaning.")
if summary["out_of_range_lat"] or summary["out_of_range_lon"]:
    issues.append("Detected coordinates outside the expected ±90°/±180° bounds.")
if issues:
    print("\n⚠️  Issues detected:")
    for item in issues:
        print("   •", item)

valid_mask = lat_numeric.notna() & lon_numeric.notna()
if valid_mask.any():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(lat_numeric[valid_mask], bins=30, color="#2D9CDB", alpha=0.85)
    axes[0].set_title("Latitude Distribution")
    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Count")

    axes[1].hist(lon_numeric[valid_mask], bins=30, color="#6C5CE7", alpha=0.85)
    axes[1].set_title("Longitude Distribution")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
else:
    print("\nNo valid coordinates to plot after cleaning.")
"""

        return template.replace("__LATITUDE_COL__", repr(latitude)).replace("__LONGITUDE_COL__", repr(longitude))


def get_component():
    return SpatialDataQualityAnalysis
