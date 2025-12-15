from typing import Dict

ROAD_FACTOR_COLUMNS = [
	"SPEED_LIMIT_MEAN",
	"LANE_COUNT_MEAN",
	"WET_ROAD_RATE",
	"ICY_ROAD_RATE",
	"INTERSECTION_RATE",
	"WORK_ZONE_RATE",
	"STATE_ROAD_RATE",
	"TURNPIKE_RATE",
]


def readable_label(col: str, label_map: Dict[str, str]) -> str:
	"""Return a friendlier label for plotting if available."""
	return label_map.get(col, col.replace("_", " ").title())
