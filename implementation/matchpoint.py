import cv2
import time
import numpy as np

import globals

from trace_match import TraceMatch
from tracking import MedianFlowTracker, initialise_roi
from widgets import Orbit
from menus import TargetMenu, DragMenu

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

    def __init__(self):
        self.state = 'matching'
        self.trace_match = TraceMatch()
        self.frame_rate = 30.0
        self.frame_time = time.time()
        self.prev_gray = None
        self.debug = False

        # Interaction menus/orbits
        self.orbits = [
            (Orbit(1 * CAM_W//4, CAM_H//2), TargetMenu(CAM_W, CAM_H)),
            (Orbit(3 * CAM_W//4, CAM_H//2, inverted=True), DragMenu(CAM_W, CAM_H))
        ]

    def _draw_info(self, frame: np.ndarray, text: str):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, globals.FONT_SIZE, 1)[0][0]
        cv2.putText(frame, text, ((CAM_W - tw) // 2, CAM_H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, globals.FONT_SIZE, globals.C_DIM, 1, cv2.LINE_AA)

    def reset(self):
        self.state        = 'matching'
        self.roi_tracker  = None
        self.cursor       = None

        if self.menu is not None:
            self.menu.reset()
            self.menu = None

        self.trace_match = TraceMatch()
        print("[reset]  back to matching state")
        pass

    def _run_matching(self, gray: np.ndarray, frame: np.ndarray):
        # Track features
        candidates = self.trace_match.track_features(gray)

        # Draw orbits + bottom text
        for orbit, _ in self.orbits:
            orbit.draw(frame)
            
        self._draw_info(frame,
            "Mirror the orbiting dot with any body movement to acquire a pointer")

        # Draw features in Debug Mode
        if (self.debug):
            self.trace_match.draw_features(frame, candidates)

        # Find the features/circles that matche
        # the motion of the orbits
        matches = self.trace_match.find_matches(candidates, [orbit for orbit, _ in self.orbits], self.frame_rate, self.frame_time)
        if not matches or self.prev_gray is None:
            return frame

        # Take the feature/match with the smallest fit error 
        m = min(matches, key=lambda m: m['circle'][2])
        tid    = m['tid']
        hist   = m['history']
        circle = m['circle']
        _, _, r_fit, _ = circle
        
        # Get Region of Interest from match
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
        self.cursor = (int(start[0]), int(start[1]))

        # Initialize MedianFlowTracker
        self.roi_tracker = MedianFlowTracker(
            roi, cd_gain, gray, CAM_W, CAM_H, start)

        # Initialize menu
        self.menu = self.orbits[tid][1]
        
        self.state  = 'coupled'
        print(f"[coupled]  roi={roi}  r={r_fit:.1f}px  cd_gain={cd_gain:.2f}")

    def _run_coupled(self, gray: np.ndarray, frame: np.ndarray):
        # Reset if Tracker or menu are missing/inactive
        if self.roi_tracker is None or self.menu is None or not self.menu.is_active():
            self.reset()
            return

        # Update cursor based on tracker
        cx, cy = self.roi_tracker.update(gray, CAM_W, CAM_H)
        self.cursor = (int(cx), int(cy))

        # Handle menu of orbit
        self.menu.update(self.cursor)
        self.menu.draw(frame)

        # Cursor crosshair
        cv2.circle(frame, self.cursor, 16, globals.C_CURSOR,    -1, cv2.LINE_AA)
        cv2.circle(frame, self.cursor, 20, globals.C_CURSOR_RIM,  2, cv2.LINE_AA)

        # Draw Region of Interest in Debug Mode
        if self.debug:
            rx, ry, rw, rh = self.roi_tracker.roi
            p1 = (int(rx), int(ry))
            p2 = (int(rx + rw), int(ry + rh))
            frame = cv2.rectangle(frame, p1, p2, globals.C_ROI_DBG, 1)

        # Draw info text
        self._draw_info(frame,
            "Move to control cursor  |  Hover a target to select  |  Still = release")

        # Reset/Decouple on idle
        if self.roi_tracker.is_idle():
            print("[decoupled]  idle timeout")
            self.reset()

    def run(self, frame: np.ndarray):
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # Make image 20% less bright
        cv2.multiply(frame, 0.2, dst=frame)

        # Run matching/coupled state logic
        match self.state:
            case 'matching':
                self._run_matching(gray, frame)       
            case _:
                self._run_coupled(gray, frame)

        # Calculate smoothed framerate/frametime
        current_frame_time = time.time()
        frame_delta = current_frame_time - self.frame_time
        self.frame_rate = (1 / frame_delta if frame_delta > 0 else 1/30) + 0.1 * self.frame_rate
        self.frame_time = current_frame_time
        self.prev_gray = gray

    def draw_hud(self, frame: np.ndarray):
        # Draw "Matchpoint | {STATE} | {FPS} HUD"
        label = "MATCHING" if self.state == 'matching' else "COUPLED "
        txt   = f"MatchPoint  |  {label}  |  {self.frame_rate:.0f} fps"
        cv2.putText(frame, txt, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, globals.C_GREY, 2, cv2.LINE_AA)
        
        # Draw Debug indicator
        if self.debug:
            cv2.putText(frame, "DEBUG", (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, globals.C_ORBIT_DOT, 1, cv2.LINE_AA)

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

    # Run matchpoint algorithm
    matchpoint.run(frame)
    matchpoint.draw_hud(frame)

    # Show application state
    cv2.imshow("MatchPoint", frame)

# Release Resource upon exit
cap.release()
cv2.destroyAllWindows()
