"""Cosmetic polish for PyBullet's default ground/lighting/viewport.

PyFlyt's Aviary IS a pybullet_utils.bullet_client.BulletClient, so once a gym
env has been reset, `env.unwrapped.env` is that live pybullet client and
exposes every raw pybullet call (loadTexture, changeVisualShape,
configureDebugVisualizer, resetDebugVisualizerCamera, ...). The ground plane's
body id is stored as `aviary.planeId`.

Ceiling: this is still PyBullet's OpenGL debug renderer, not a game engine or
ray tracer. No PBR materials, no dynamic sky/weather, no real global
illumination. This module only swaps the flat grey plane for a checker
texture, turns on shadows, hides GUI clutter, and sets a nicer default camera.
"""
from __future__ import annotations


def apply_terrain_polish(env) -> None:
    """Best-effort cosmetic polish. Never raises: every pybullet call is
    wrapped so an unsupported flag/method in this pybullet build, or the
    absence of a GUI connection (headless), just gets skipped.
    """
    aviary = getattr(getattr(env, "unwrapped", env), "env", None)
    if aviary is None:
        return

    # Ground texture: swap flat grey plane for pybullet's built-in checker.
    try:
        import pybullet_data

        tex_path = pybullet_data.getDataPath() + "/checker_grid.jpg"
        tex_id = aviary.loadTexture(tex_path)
        aviary.changeVisualShape(aviary.planeId, -1, textureUniqueId=tex_id)
    except Exception:
        pass

    # Debug visualizer flags (each independent, GUI-only, no-op headless).
    for flag_name, value in (
        ("COV_ENABLE_GUI", 0),
        ("COV_ENABLE_SHADOWS", 1),
        ("COV_ENABLE_MOUSE_PICKING", 0),
    ):
        try:
            flag = getattr(aviary, flag_name, None)
            if flag is not None:
                aviary.configureDebugVisualizer(flag, value)
        except Exception:
            pass

    # A nicer default framing than pybullet's zoomed-out-at-origin default.
    try:
        aviary.resetDebugVisualizerCamera(
            cameraDistance=8,
            cameraYaw=50,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
        )
    except Exception:
        pass
