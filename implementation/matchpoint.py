import cv2
import time
import numpy as np

from trace_match import TraceMatch
from widgets import Orbit, Target
from tracking import MedianFlowTracker, initialise_roi

import globals

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (device 0).")

cap.set(cv2.CAP_PROP_FPS, 30)
CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("-" * 60)
print("MatchPoint  |  UIST 2017 replication")
print("Mirror the orbiting dot to acquire a touchless cursor.")
print("Controls:  d = debug overlay   r = reset   q = quit")
print("-" * 60)

class Matchpoint:

    def __init__(self, matching_widget = None):
        self.state = 'matching'
        self.matching_widget = matching_widget
        self.trace_match = TraceMatch()
        self.orbit = Orbit(CAM_W//2, CAM_H//2)
        self.frame_rate = 30.0
        self.frame_time = time.time()
        self.prev_gray = None
        self.debug = False

        # TODO: Replace with more general method
        self.targets   = self._make_targets()

    def _make_targets(self) -> list:
        cx, cy = CAM_W // 2, CAM_H // 2
        R      = 120
        return [Target(int(cx + R * np.cos(i * np.pi / 2)),
                       int(cy + R * np.sin(i * np.pi / 2)),
                       r=45, label=lbl)
                for i, lbl in enumerate(["A", "B", "C", "D"])]

    # TODO: Replace
    def _put_hint(self, canvas: np.ndarray, text: str):
        fs = 0.55
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
        cv2.putText(canvas, text, ((CAM_W - tw) // 2, CAM_H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, globals.C_DIM, 1, cv2.LINE_AA)

    def reset(self):
        self.state        = 'matching'
        self.roi_tracker  = None
        self.cursor       = None
        self.trace_match = TraceMatch()
        for t in self.targets:
            t.dwell_t0  = None
            t.activated = False
        print("[reset]  back to matching state")
        pass

    def _run_matching(self, gray: np.ndarray, frame: np.ndarray):
        # Track features
        candidates = self.trace_match.track_features(gray)
        self.orbit.draw(frame)
        self._put_hint(frame,
            "Mirror the orbiting dot with any body movement to acquire a pointer")

        if (self.debug):
            frame = self.trace_match.draw_features(frame, candidates)

        # Find matches using pearson and ransac
        matches = self.trace_match.find_matches(candidates, [self.orbit], self.frame_rate, self.frame_time)
        if not matches or self.prev_gray is None:
            return frame

        m = matches[0]   # take first (all are valid; one is sufficient)
        hist   = m['history']
        circle = m['circle']
        _, _, r_fit, _ = circle
        
        roi = initialise_roi(self.prev_gray, gray, [hist], CAM_W, CAM_H)
        if roi is None:
            return

        # CD-gain  (Eq. 8):  CDgain = (1/r) × screen_dim
        cd_gain = (1.0 / max(r_fit, 4.0)) * max(CAM_W, CAM_H)

        # Cursor starts at the mapped position of the matched feature
        roi_cx = roi[0] + roi[2] / 2.0
        roi_cy = roi[1] + roi[3] / 2.0
        start  = (float(np.clip(roi_cx, 0, CAM_W - 1)),
                  float(np.clip(roi_cy, 0, CAM_H - 1)))

        self.roi_tracker = MedianFlowTracker(
            roi, cd_gain, gray, CAM_W, CAM_H, start)
        self.cursor = (int(start[0]), int(start[1]))
        self.state  = 'coupled'
        print(f"[coupled]  roi={roi}  r={r_fit:.1f}px  cd_gain={cd_gain:.2f}")

        return frame

    def _run_coupled(self, gray: np.ndarray, frame: np.ndarray):
        if self.roi_tracker is None:
            self.reset()
            return frame

        cx, cy = self.roi_tracker.update(gray, CAM_W, CAM_H)
        self.cursor = (int(cx), int(cy))

        for tgt in self.targets:
            tgt.update(self.cursor)
            tgt.draw(frame)

        # Cursor crosshair
        frame = cv2.circle(frame, self.cursor, 16, globals.C_CURSOR,    -1, cv2.LINE_AA)
        frame = cv2.circle(frame, self.cursor, 20, globals.C_CURSOR_RIM,  2, cv2.LINE_AA)

        if self.debug:
            rx, ry, rw, rh = self.roi_tracker.roi
            p1 = (int(rx), int(ry))
            p2 = (int(rx + rw), int(ry + rh))
            frame = cv2.rectangle(frame, p1, p2, globals.C_ROI_DBG, 1)

        self._put_hint(frame,
            "Move to control cursor  |  Hover a target to select  |  Still = release")

        if self.roi_tracker.is_idle():
            print("[decoupled]  idle timeout")
            self.reset()

        return frame

    def run(self, frame: np.ndarray):
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # Make image 20% less bright
        frame = cv2.multiply(frame, 0.2)

        match self.state:
            case 'matching':
                frame = self._run_matching(gray, frame)       
            case _:
                frame = self._run_coupled(gray, frame)

        # Calculate soomthed framerate/frametime
        current_frame_time = time.time()
        frame_delta = current_frame_time - self.frame_time
        self.frame_rate = (1 / frame_delta if frame_delta > 0 else 1/30) + 0.1 * self.frame_rate
        self.frame_time = current_frame_time
        self.prev_gray = gray

        return frame

    def draw_hud(self, frame: np.ndarray):
        label = "MATCHING" if self.state == 'matching' else "COUPLED "
        txt   = f"MatchPoint  |  {label}  |  {self.frame_rate:.0f} fps"
        frame = cv2.putText(frame, txt, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, globals.C_GREY, 2, cv2.LINE_AA)
        if self.debug:
            frame = cv2.putText(frame, "DEBUG", (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, globals.C_ORBIT_DOT, 1, cv2.LINE_AA)

        return frame

matchpoint = Matchpoint()

while True:
    match cv2.waitKey(1):
        # Enable Reset Matchpoint
        case 114: # ASCII Code of r
            matchpoint.reset()
        # Enable Debug-Mode
        case 100: # ASCII Code of d
            matchpoint.debug = not matchpoint.debug
        # Exit Application
        case 113: # ASCII Code of q
            break

    ret, frame = cap.read()

    if not ret:
        continue

    # mirror for natural use
    frame = cv2.flip(frame, 1)

    # Do the background thingy

    frame = matchpoint.run(frame)

    frame = matchpoint.draw_hud(frame)
    cv2.imshow("MatchPoint", frame)

cap.release()
cv2.destroyAllWindows()
