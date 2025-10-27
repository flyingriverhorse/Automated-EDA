"""Geospatial projection validation component."""

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


class CoordinateSystemProjectionCheck:
	"""Surface projection and coordinate quality checks for geospatial data."""

	def get_metadata(self) -> Dict[str, Any]:
		return {
			"name": "Coordinate System & Projection Check",
			"description": "Validate latitude/longitude ranges and flag potential projection issues.",
			"category": "Geospatial Analysis",
			"complexity": "beginner",
			"estimated_runtime": "1-3 seconds",
			"tags": ["geospatial", "quality", "projection", "validation"],
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

		return f"""
# COORDINATE SYSTEM & PROJECTION CHECK
import numpy as np

required_columns = [{latitude!r}, {longitude!r}]
missing = [col for col in required_columns if col not in df.columns]
if missing:
	raise ValueError(f"Missing required coordinate columns: {{missing}}")

geo_df = df[[{latitude!r}, {longitude!r}]].dropna()
geo_df = geo_df.apply(pd.to_numeric, errors='coerce').dropna()

if geo_df.empty:
	raise ValueError("No usable latitude/longitude pairs after coercion.")

lat_min, lat_max = geo_df[{latitude!r}].min(), geo_df[{latitude!r}].max()
lon_min, lon_max = geo_df[{longitude!r}].min(), geo_df[{longitude!r}].max()

print("🧭 Coordinate System & Projection Check")
print("• Points analysed:", len(geo_df))
print("• Latitude range:", f"{{lat_min:.4f}} → {{lat_max:.4f}}")
print("• Longitude range:", f"{{lon_min:.4f}} → {{lon_max:.4f}}")

issues = []
if np.abs(geo_df[{latitude!r}]).max() > 90:
	issues.append("Latitude values exceed ±90°. Data may be stored in a projected CRS.")
if np.abs(geo_df[{longitude!r}]).max() > 180:
	issues.append("Longitude values exceed ±180°. Confirm decimal degree storage (EPSG:4326).")
if lat_max - lat_min == 0 or lon_max - lon_min == 0:
	issues.append("Latitude or longitude shows zero spread. Coordinates may be constant.")

if issues:
	print("\n⚠️  Potential issues detected:")
	for item in issues:
		print(" •", item)
else:
	print("\n✅ Coordinate ranges fall within expected decimal-degree bounds.")
"""


def get_component():
	return CoordinateSystemProjectionCheck
