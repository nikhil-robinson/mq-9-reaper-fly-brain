"""Live on-screen flight HUD via PyBullet debug text (read-only consumer of
course.py / motor.py outputs — no physics or brain logic here)."""

import math


class FlightHUD:
    """Live on-screen readout via PyBullet debug text, updated in place each step.

    No-op if the env has no real PyBullet GUI (render_mode != "human").
    """

    def __init__(self, env):
        self.render_mode = getattr(env.unwrapped, "render_mode", None)
        self.client = getattr(env.unwrapped, "env", None) if self.render_mode == "human" else None
        self._text_id = None

    def update(self, position, bearing_error, distance, rates, action):
        if self.client is None:
            return

        bearing_deg = math.degrees(bearing_error)
        text_pos = [position[0], position[1], position[2] + 3.0]
        lines = [
            f"bearing: {bearing_deg:+.1f}deg  dist: {distance:.1f}m",
            "  ".join(f"{name}: {rate:.2f}" for name, rate in rates.items()),
            "aileron: {:.2f}  elevator: {:.2f}  rudder: {:.2f}  throttle: {:.2f}".format(*action),
        ]
        text = "\n".join(lines)

        try:
            kwargs = {}
            if self._text_id is not None:
                kwargs["replaceItemUniqueId"] = self._text_id
            self._text_id = self.client.addUserDebugText(
                text,
                text_pos,
                textColorRGB=[1, 1, 0],
                textSize=1.2,
                lifeTime=0,
                **kwargs,
            )
        except Exception as exc:
            # No live GUI connection (e.g. connection dropped) — degrade to no-op.
            if "Not connected" in str(exc) or "connection" in str(exc).lower():
                self.client = None
            else:
                raise
