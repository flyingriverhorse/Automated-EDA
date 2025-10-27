"""Spatial relationships analysis component."""

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


class SpatialRelationshipsAnalysis:
	"""Geospatial proximity analysis metadata and notebook snippet."""

	def get_metadata(self) -> Dict[str, Any]:
		return {
			"name": "Spatial Relationships Analysis",
			"description": "Compute nearest-neighbour distances to understand spatial clustering.",
			"category": "Geospatial Analysis",
			"complexity": "intermediate",
			"estimated_runtime": "4-8 seconds",
			"tags": ["geospatial", "distance", "clustering", "relationships"],
			"icon": "📐",
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
# SPATIAL RELATIONSHIPS ANALYSIS
import numpy as np
import matplotlib.pyplot as plt

required = [{latitude!r}, {longitude!r}]
missing = [col for col in required if col not in df.columns]
if missing:
	raise ValueError(f"Missing required coordinate columns: {{missing}}")

geo_df = df[required].dropna()
geo_df = geo_df.apply(pd.to_numeric, errors='coerce').dropna()

if len(geo_df) < 2:
	raise ValueError("Need at least two observations with latitude/longitude values.")

sample = geo_df.sample(n=min(len(geo_df), 250), random_state=42)

def haversine(lat1, lon1, lat2, lon2):
	R = 6371.0  # Earth radius in kilometres
	dlat = np.radians(lat2 - lat1)
	dlon = np.radians(lon2 - lon1)
	a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	return R * c

coords = sample[{latitude!r}].to_numpy(), sample[{longitude!r}].to_numpy()
latitudes, longitudes = coords

dist_matrix = np.zeros((len(sample), len(sample)))
for i in range(len(sample)):
	for j in range(i + 1, len(sample)):
		distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
		dist_matrix[i, j] = distance
		dist_matrix[j, i] = distance

nearest = []
for i in range(len(sample)):
	row = dist_matrix[i]
	row[row == 0] = np.nan
	nearest.append(np.nanmin(row))

nearest = np.array(nearest)

print("📐 Spatial Relationships")
print("Rows analysed:", len(geo_df))
print("Sample size:", len(sample))
print("Mean nearest neighbour (km):", np.nanmean(nearest))
print("Median nearest neighbour (km):", np.nanmedian(nearest))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(nearest[~np.isnan(nearest)], bins=20, color="#6C5CE7", alpha=0.85)
ax.set_xlabel("Nearest neighbour distance (km)")
ax.set_ylabel("Count")
ax.set_title("Distribution of nearest neighbour distances")
plt.show()
"""


def get_component():
	return SpatialRelationshipsAnalysis
