from pathlib import Path

from fly_brain.mq9_skin import mq9_drone_options


def test_real_skin_dir_returns_model_dir():
    opts = mq9_drone_options()
    assert "model_dir" in opts
    model_dir = Path(opts["model_dir"])
    assert (model_dir / "fixedwing" / "fixedwing.urdf").is_file()
    assert (model_dir / "fixedwing" / "fixedwing.yaml").is_file()


def test_missing_skin_dir_falls_back_to_empty_dict():
    assert mq9_drone_options(skin_dir="/nonexistent/path") == {}
