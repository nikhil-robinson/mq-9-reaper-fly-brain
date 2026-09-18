"""Points PyFlyt at the MQ-9 cosmetic skin, falling back to its stock mesh."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SKIN_DIR = _REPO_ROOT / "assets" / "vehicles" / "mq9"


def mq9_drone_options(skin_dir=None):
    """Returns a `drone_options` dict for PyFlyt's Aviary that points at the
    MQ-9 skin directory (default: assets/vehicles/mq9 relative to repo root)
    if it exists, else returns {} (falls back to PyFlyt's stock mesh)."""
    model_dir = Path(skin_dir) if skin_dir is not None else _DEFAULT_SKIN_DIR
    urdf = model_dir / "fixedwing" / "fixedwing.urdf"
    yaml_file = model_dir / "fixedwing" / "fixedwing.yaml"
    if urdf.is_file() and yaml_file.is_file():
        return {"model_dir": str(model_dir)}
    return {}
