import cv2
import time
import numpy as np
from collections import deque
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

# Import globals
from globals import DEDUPE_WIN, MOTION_THRESHOLD, PEARSON_THRESH, MIN_ARC
from globals import RANSAC_N_ITER, RANSAC_MIN_IN, MIN_DISP_PX, THIN

def PCC(feature, target):
    f_x, f_y = (feature[:, 0], feature[:, 1])
    t_x, t_y = (target[:, 0], target[:, 1])
    corrx, _ = pearsonr(f_x, t_x)
    corry, _ = pearsonr(f_y, t_y)
    return corrx, corry

def fit_circle_ransac(pts: np.ndarray):
    """
    Fit a circle to 2-D points using RANSAC.
    Returns (cx, cy, r, n_inliers) or None if no robust fit found.
    The paper (§ "Motion-Matching") checks arc length after this fit.
    """
    if len(pts) < 3:
        return None

    best, best_n = None, 0

    for _ in range(RANSAC_N_ITER):
        idx = np.random.choice(len(pts), 3, replace=False)
        ax, ay = pts[idx[0]]
        bx, by = pts[idx[1]]
        cx_, cy_ = pts[idx[2]]

        D = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(D) < 1e-7:
            continue

        ux = ((ax*ax + ay*ay) * (by - cy_) +
              (bx*bx + by*by) * (cy_ - ay) +
              (cx_*cx_ + cy_*cy_) * (ay - by)) / D
        uy = ((ax*ax + ay*ay) * (cx_ - bx) +
              (bx*bx + by*by) * (ax - cx_) +
              (cx_*cx_ + cy_*cy_) * (bx - ax)) / D
        r  = np.sqrt((ax - ux)**2 + (ay - uy)**2)

        if r < 5.0 or r > 600.0:
            continue

        dist = np.sqrt((pts[:, 0] - ux)**2 + (pts[:, 1] - uy)**2)
        inliers = (1 - THIN) * r < dist
        inliers &= dist < (1 + THIN) * r
        n_in = int(np.sum(inliers))

        if n_in > best_n:
            best_n, best = n_in, (ux, uy, r, n_in)

    if best is None or best[3] < RANSAC_MIN_IN:
        return None
    return best

def arc_coverage(pts: np.ndarray, cx: float, cy: float) -> float:
    """
    Estimate the fraction of the circle arc covered by pts.
    Discretises angles into 16 bins; returns (non-empty bins) / 16.
    """
    if len(pts) < 2:
        return 0.0
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins   = np.floor((angles + np.pi) / (2.0 * np.pi) * 16).astype(int) % 16
    return len(np.unique(bins)) / 16.0

class TraceMatch:

    def __init__(self, W=10, min_displacement=4):
        self.W = W
        self.min_displacement = min_displacement

        # Values used in the TraceMatch Paper
        self._fast = cv2.FastFeatureDetector_create(
            threshold=20, nonmaxSuppression=True)
        self._LK = dict(winSize=(51, 51), maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.prev_pts = None
        self.prev_gray = None
        self.history = []

    def track_features(self, gray):
        # Track existing points using optical flow
        if self.prev_gray is not None and len(self.history) > 0:
            new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self._LK)

            self.history = [
                (pts.append(new_pts[i]), pts)[1]
                for i, pts in enumerate(self.history)
                if st[i] == 1
            ]
            self.prev_pts = np.array([pts[-1] for pts in self.history], dtype=np.float32)

        # Detect all keypoints
        keypoints = self._fast.detect(gray, None)

        # Remove duplicates in the deduplication window
        if self.prev_pts is not None:
            tracked_pts = self.prev_pts.copy()
            unique_keypoints = []

            for kp in keypoints:
                _kp = np.array(kp.pt)

                diffs = np.abs(tracked_pts - _kp)
                if not np.any((diffs[:, 0] < DEDUPE_WIN) & (diffs[:, 1] < DEDUPE_WIN)):
                    unique_keypoints.append(kp)
                    tracked_pts = np.vstack([tracked_pts, _kp])

            keypoints = unique_keypoints

        # Add new keypoints to history
        self.history.extend([deque([np.array(kp.pt)], maxlen=self.W) for kp in keypoints])

        # Set values for next iteration
        self.prev_gray = gray
        self.prev_pts = np.array([pts[-1] for pts in self.history], dtype=np.float32)

        # Find suitable motion candidates
        candidates = []
        for i, pts in enumerate(self.history):
            if len(pts) < self.W:
                continue
            
            displacements = np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)
            avg_displacement = np.mean(displacements)
            if avg_displacement > self.min_displacement:
                candidates.append((i, pts))

        return candidates

    def draw_features(self, frame, candidates):
        candidate_i = [i for i, _ in candidates]

        for i, pt in enumerate(self.prev_pts):
            color = (255,0,0) if i in candidate_i else (0,255,0)
            frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)

        return frame

    def find_matches(self, candidates, targets, frame_rate, frame_time):
        matches = []

        for target in targets:
            target_x = target.trajectory_x(self.W, frame_rate, frame_time)
            target_y = target.trajectory_y(self.W, frame_rate, frame_time)
            target = np.array([(target_x[i], target_y[i]) for i in range(self.W)])

            for i, pts in candidates:
                pts = np.array(pts)

                # Reject barely-moving features
                if np.linalg.norm(pts) < MIN_DISP_PX:
                    continue

                corrx, corry = PCC(pts, target)
                if not (corrx > PEARSON_THRESH and corry > PEARSON_THRESH):
                    continue
            
                circle = fit_circle_ransac(pts)
                if circle is None:
                    continue
            
                cx_c, cy_c, r_c, _ = circle
                if arc_coverage(pts, cx_c, cy_c) < MIN_ARC:
                    continue
            
                matches.append({'fid': i, 'history': pts, 'circle': circle})
        return matches
