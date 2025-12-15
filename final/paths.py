from pathlib import Path

# Resolve project-level directories from inside final/
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
AGG_DIR = DATA_DIR / "aggregates"
PLOTS_DIR = ROOT_DIR / "plots"


def ensure_plots_dir() -> None:
	"""Create plots output directory if missing."""
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)
