import cv2
import math
import time
import numpy as np

from globals import ORBIT_R, ORBIT_T, ORBIT_DOT_R
from globals import C_ORBIT_RING, C_ORBIT_DOT
from globals import C_TGT_IDLE, C_TGT_DWELL, C_TGT_DONE, C_WHITE
from globals import DWELL_TIME

class Orbit:
    """
    A ring with a dot orbiting at a fixed angular velocity.

    The dot's x-position follows a cosine wave and y-position a sine wave.
    These serve as the reference signals for Pearson correlation in Phase 1.
    """

    def __init__(self, cx: int, cy: int,
                 radius: int = ORBIT_R, period: float = ORBIT_T, inverted=False):
        self.cx, self.cy = cx, cy
        self.radius      = radius
        self.period      = -period if inverted else period
        self._t0         = time.time()   # phase reference

    # ── reference signal helpers ──────────────────────────────────────────

    def _angle(self, t: float) -> float:
        return (2.0 * math.pi * (t - self._t0) / self.period) % (2.0 * math.pi)

    def ref_x(self, t: float) -> float:
        return self.cx + self.radius * math.cos(self._angle(t))

    def ref_y(self, t: float) -> float:
        return self.cy + self.radius * math.sin(self._angle(t))

    def trajectory_x(self, n: int, fps: float, t_now: float) -> np.ndarray:
        """Expected x-positions for the last n frames up to t_now (oldest first)."""
        dt = 1.0 / max(fps, 1.0)
        return np.array([self.ref_x(t_now - (n - 1 - i) * dt) for i in range(n)])

    def trajectory_y(self, n: int, fps: float, t_now: float) -> np.ndarray:
        dt = 1.0 / max(fps, 1.0)
        return np.array([self.ref_y(t_now - (n - 1 - i) * dt) for i in range(n)])

    # ── drawing ───────────────────────────────────────────────────────────

    def _dot_pos(self) -> tuple:
        a = self._angle(time.time())
        return (int(self.cx + self.radius * math.cos(a)),
                int(self.cy + self.radius * math.sin(a)))

    def draw(self, frame: np.ndarray):
        # outer ring
        cv2.circle(frame, (self.cx, self.cy),
                   self.radius, C_ORBIT_RING, 2, cv2.LINE_AA)
        # centre marker
        cv2.circle(frame, (self.cx, self.cy), 5, C_ORBIT_RING, -1, cv2.LINE_AA)
        # orbiting dot
        cv2.circle(frame, self._dot_pos(),
                   ORBIT_DOT_R, C_ORBIT_DOT, -1, cv2.LINE_AA)

class Target:
    """
    A circular target that activates after the cursor dwells on it for DWELL_TIME.
    Provides a concrete demonstration of the cursor interaction.
    """

    def __init__(self, cx: int, cy: int, r: int = 45, label: str = ""):
        self.cx, self.cy = cx, cy
        self.r           = r
        self.label       = label
        self.dwell_t0    = None
        self.activated   = False

    def update(self, cursor):
        if cursor is None:
            self.dwell_t0 = None
            return
        if math.hypot(cursor[0] - self.cx, cursor[1] - self.cy) <= self.r:
            if self.dwell_t0 is None:
                self.dwell_t0 = time.time()
            elif not self.activated and (time.time() - self.dwell_t0) >= DWELL_TIME:
                self.activated = True
        else:
            self.dwell_t0 = None

    def draw(self, frame: np.ndarray):
        if self.activated:
            cv2.circle(frame, (self.cx, self.cy), self.r, C_TGT_DONE, -1, cv2.LINE_AA)
            cv2.circle(frame, (self.cx, self.cy), self.r, C_WHITE,    2,  cv2.LINE_AA)
        else:
            frac = 0.0
            if self.dwell_t0 is not None:
                frac = min(1.0, (time.time() - self.dwell_t0) / DWELL_TIME)
            col = C_TGT_DWELL if frac > 0 else C_TGT_IDLE
            cv2.circle(frame, (self.cx, self.cy), self.r, col, 2, cv2.LINE_AA)
            if frac > 0:
                cv2.ellipse(frame, (self.cx, self.cy),
                            (self.r + 8, self.r + 8), -90, 0, int(frac * 360),
                            C_TGT_DWELL, 5, cv2.LINE_AA)

        if self.label:
            fs   = 0.75
            size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
            tx   = self.cx - size[0] // 2
            ty   = self.cy + size[1] // 2
            cv2.putText(frame, self.label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, C_WHITE, 2, cv2.LINE_AA)

