#!/usr/bin/env python3
"""
MatchPoint – Spontaneous Spatial Coupling of Body Movement for Touchless Pointing
Replication of: Clarke & Gellersen, UIST 2017
Paper: https://eprints.lancs.ac.uk/id/eprint/88665/

─── How it works ────────────────────────────────────────────────────────────
  An Orbit widget (a ring with a dot orbiting around it at a fixed period) is
  displayed on screen.  The user mirrors the dot's circular motion with any
  part of their body (hand, head, object …).

  Phase 1 – Motion Matching (§ "Motion-Matching" in the paper)
    • Detect FAST feature points in each webcam frame.
    • Track them frame-to-frame with pyramidal Lucas-Kanade optical flow.
    • For every feature that has accumulated HIST_LEN frames of trajectory:
        – Compute Pearson r between the feature's x-trajectory and the Orbit's
          expected cosine signal, and between y-trajectory and the sine signal.
        – If max(|rx|, |ry|) ≥ PEARSON_THRESH AND a circle can be RANSAC-fitted
          to the trajectory (covering ≥ MIN_ARC of its circumference) → MATCH.

  Phase 2 – ROI Initialisation (§ "Tracker Initialisation" in the paper)
    • Compute dense Farnebäck optical flow between the last two frames.
    • Build a pixel mask: keep pixels whose flow magnitude is within TH_D of
      the matched feature's displacement magnitude, and whose angle is within
      TH_ANGL of the average trajectory angle  (Eq. 6–7 in the paper).
    • Connected-component labelling seeded from the matched feature point(s).
    • Fit a bounding-box → initial ROI.
    • Compute CD-gain = (1 / r_fit) × screen_dim  (Eq. 8 in the paper).

  Phase 3 – Tracking (§ "Tracking" in the paper, modified Median Flow)
    • Generate a grid of tracking points inside the ROI using spacing inspired
      by the central polygonal / lazy caterer's sequence (paper reference).
    • Track with Lucas-Kanade; take the *median* displacement (Median Flow).
    • Recalibrate the ROI every RECALIB_INT seconds (when ROI moved enough)
      using dense OF; record an offset so the cursor does not jump.
    • Jitter filter: dynamic moving-window average when motion < JITTER_DMIN
      (Eq. 9 in the paper).
    • Map ROI centre → cursor via CD-gain (relative to the initial position).

  Pointer Termination
    • Cursor decouples automatically after IDLE_TIMEOUT seconds of stillness.

─── Controls ────────────────────────────────────────────────────────────────
  d  – toggle debug overlay (feature points, ROI box)
  r  – reset to matching state
  q  – quit

─── Usage ───────────────────────────────────────────────────────────────────
  pip install -r requirements.txt
  python3 matchpoint.py
"""

import math
import time
from collections import deque

import cv2
import numpy as np
from scipy.stats import pearsonr

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1280, 720       # display window size (px)
CAM_W, CAM_H = 640,  480      # webcam capture resolution (px)

# Camera → display scale factors
SCALE_X = WIN_W / CAM_W       # 2.0
SCALE_Y = WIN_H / CAM_H       # 1.5

# ── Orbit widget ──────────────────────────────────────────────────────────
ORBIT_R     = 55              # radius of the orbit path (px, display coords)
ORBIT_DOT_R = 13              # radius of the orbiting dot (px)
ORBIT_T     = 2.5             # revolution period (seconds)

# ── Phase 1 : Motion Matching ─────────────────────────────────────────────
HIST_LEN       = 38           # frames of trajectory history (≈ 1.25 s @ 30 fps)
MIN_DISP_PX    = 2.0          # minimum total displacement to consider a feature (px)
PEARSON_THRESH = 0.75         # |Pearson r| threshold for accepting a match
RANSAC_N_ITER  = 120          # RANSAC iterations for circle fit
RANSAC_EPS     = 8.0          # inlier distance threshold (px)
RANSAC_MIN_IN  = 6            # minimum inliers to accept a circle fit
MIN_ARC        = 0.22         # minimum fraction of circle arc that must be covered

# ── Phase 2 : ROI Initialisation ─────────────────────────────────────────
TH_D    = 0.25                # distance tolerance for dense-OF pixel match (Eq. 6)
TH_ANGL = math.pi / 8        # angle tolerance ≈ 22.5°                     (Eq. 7)
MIN_ROI = 30                  # fallback minimum ROI side length (px)

# ── Phase 3 : Modified Median Flow ───────────────────────────────────────
RECALIB_MOVE = 2.0            # minimum ROI displacement to trigger recalibration (px)
RECALIB_INT  = 0.5            # minimum seconds between recalibrations
JITTER_DMIN  = 2.5            # motion below this activates jitter filter (px)  [paper: 2.5]
JITTER_NMAX  = 10             # maximum moving-window size for jitter filter      [paper: 10]

# ── Pointer Termination ───────────────────────────────────────────────────
IDLE_TIMEOUT = 3.0            # seconds of stillness → decouple

# ── Dwell Selection ───────────────────────────────────────────────────────
DWELL_TIME = 0.8              # seconds to dwell over a target to activate it

# ── Colours (BGR) ─────────────────────────────────────────────────────────
C_ORBIT_RING = ( 55, 185,  55)
C_ORBIT_DOT  = ( 20, 235,  20)
C_CURSOR     = (  0, 215, 255)
C_CURSOR_RIM = (  0, 140, 200)
C_TGT_IDLE   = ( 55,  55, 205)
C_TGT_DWELL  = (  0, 185, 255)
C_TGT_DONE   = (  0, 240,  80)
C_ROI_DBG    = (255,  55,  55)
C_FEAT_DBG   = (  0, 255,   0)
C_WHITE      = (255, 255, 255)
C_GREY       = (175, 175, 175)
C_DIM        = (105, 105, 105)


# ═════════════════════════════════════════════════════════════════════════════
#  ORBIT WIDGET
# ═════════════════════════════════════════════════════════════════════════════

class Orbit:
    """
    A ring with a dot orbiting at a fixed angular velocity.

    The dot's x-position follows a cosine wave and y-position a sine wave.
    These serve as the reference signals for Pearson correlation in Phase 1.
    """

    def __init__(self, cx: int, cy: int,
                 radius: int = ORBIT_R, period: float = ORBIT_T):
        self.cx, self.cy = cx, cy
        self.radius      = radius
        self.period      = period
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

    def draw(self, canvas: np.ndarray):
        # outer ring
        cv2.circle(canvas, (self.cx, self.cy),
                   self.radius, C_ORBIT_RING, 2, cv2.LINE_AA)
        # centre marker
        cv2.circle(canvas, (self.cx, self.cy), 5, C_ORBIT_RING, -1, cv2.LINE_AA)
        # orbiting dot
        cv2.circle(canvas, self._dot_pos(),
                   ORBIT_DOT_R, C_ORBIT_DOT, -1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  RANSAC CIRCLE FIT  (used in Phase 1 to verify trajectory shape)
# ═════════════════════════════════════════════════════════════════════════════

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
        r  = math.sqrt((ax - ux)**2 + (ay - uy)**2)

        if r < 5.0 or r > 600.0:
            continue

        dists = np.abs(
            np.sqrt((pts[:, 0] - ux)**2 + (pts[:, 1] - uy)**2) - r)
        n_in = int(np.sum(dists < RANSAC_EPS))

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
    bins   = np.floor((angles + math.pi) / (2.0 * math.pi) * 16).astype(int) % 16
    return len(np.unique(bins)) / 16.0


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE TRACKER  (FAST + pyramidal Lucas-Kanade optical flow)
# ═════════════════════════════════════════════════════════════════════════════

class FeatureTracker:
    """
    Maintains FAST feature points and a fixed-length trajectory history per point.
    Points are propagated frame-to-frame via pyramidal LK optical flow.
    New points are detected when the count drops below a threshold or on schedule.
    """

    _LK = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self):
        self._fast   = cv2.FastFeatureDetector_create(
            threshold=20, nonmaxSuppression=True)
        self._pts    = None    # (N, 1, 2) float32 – current positions
        self._ids    = []      # parallel list of integer IDs
        self._hist   = {}      # id → deque of (x, y)
        self._nid    = 0       # next unique feature ID
        self._prev_g = None
        self._frame  = 0

    def update(self, gray: np.ndarray):
        """Call every frame with the current (already-mirrored) grayscale image."""
        self._frame += 1

        if self._prev_g is None:
            self._prev_g = gray
            self._detect(gray)
            return

        # ── propagate existing points ─────────────────────────────────────
        if self._pts is not None and len(self._pts) > 0:
            new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_g, gray, self._pts, None, **self._LK)

            kept_pts, kept_ids = [], []
            for i, (s, npt) in enumerate(zip(st.ravel(), new_pts)):
                if s != 1:
                    continue
                fid    = self._ids[i]
                x, y   = float(npt[0][0]), float(npt[0][1])
                if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
                    continue
                self._hist[fid].append((x, y))
                kept_pts.append(npt)
                kept_ids.append(fid)

            gone = set(self._ids) - set(kept_ids)
            for g in gone:
                self._hist.pop(g, None)

            self._pts = (np.array(kept_pts, dtype=np.float32)
                         if kept_pts else None)
            self._ids = kept_ids

        # ── re-detect when sparse or on schedule ─────────────────────────
        if len(self._ids) < 60 or self._frame % 15 == 0:
            self._detect(gray)

        self._prev_g = gray

    def _detect(self, gray: np.ndarray):
        kps = self._fast.detect(gray, None)
        if not kps:
            return

        # Occupancy grid to avoid crowding
        grid = {}
        if self._pts is not None:
            for pt in self._pts:
                grid[(int(pt[0][0] / 12), int(pt[0][1] / 12))] = True

        new_arr, new_ids = [], []
        for kp in kps:
            cell = (int(kp.pt[0] / 12), int(kp.pt[1] / 12))
            if cell in grid:
                continue
            new_arr.append([[kp.pt[0], kp.pt[1]]])
            h = deque(maxlen=HIST_LEN)
            h.append(kp.pt)
            self._hist[self._nid] = h
            new_ids.append(self._nid)
            self._nid += 1
            grid[cell] = True

        if not new_arr:
            return

        arr = np.array(new_arr, dtype=np.float32)
        if self._pts is not None:
            self._pts = np.concatenate([self._pts, arr])
            self._ids = self._ids + new_ids
        else:
            self._pts, self._ids = arr, new_ids

    def full_tracks(self) -> list:
        """Return [(id, [positions])] only for features with full HIST_LEN history."""
        return [(fid, list(self._hist[fid]))
                for fid in self._ids
                if fid in self._hist and len(self._hist[fid]) == HIST_LEN]


# ═════════════════════════════════════════════════════════════════════════════
#  MOTION MATCHER  (Phase 1 – Pearson correlation + RANSAC circle fit)
# ═════════════════════════════════════════════════════════════════════════════

class MotionMatcher:
    """
    For each tracked feature with a full trajectory history:
      1. Correlate its x-trajectory with the Orbit's cosine reference.
      2. Correlate its y-trajectory with the Orbit's sine  reference.
      3. If max(|rx|, |ry|) ≥ PEARSON_THRESH → candidate.
      4. RANSAC circle fit on the trajectory; check arc coverage ≥ MIN_ARC.
      5. Confirmed matches are returned.

    (Pearson correlation is scale-invariant, so camera-space vs display-space
    sizes do not matter — only the temporal *shape* of the motion matters.)
    """

    def __init__(self, orbit: Orbit):
        self.orbit = orbit
        self.fps   = 30.0

    def check(self, feature_tracks: list, t_now: float) -> list:
        """
        Returns a list of match dicts:
          { 'fid': int, 'history': [...], 'circle': (cx, cy, r, n_inliers) }
        """
        matches = []

        # Pre-compute and normalise orbit reference signals for this frame
        exp_x = self.orbit.trajectory_x(HIST_LEN, self.fps, t_now)
        exp_y = self.orbit.trajectory_y(HIST_LEN, self.fps, t_now)
        exp_x_n = _znorm(exp_x)
        exp_y_n = _znorm(exp_y)
        if exp_x_n is None or exp_y_n is None:
            return matches

        for fid, hist in feature_tracks:
            xs = np.array([p[0] for p in hist])
            ys = np.array([p[1] for p in hist])

            # Reject barely-moving features
            if math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < MIN_DISP_PX:
                continue

            xs_n = _znorm(xs)
            ys_n = _znorm(ys)
            if xs_n is None or ys_n is None:
                continue

            try:
                rx, _ = pearsonr(xs_n, exp_x_n)
                ry, _ = pearsonr(ys_n, exp_y_n)
            except Exception:
                continue

            if max(abs(rx), abs(ry)) < PEARSON_THRESH:
                continue

            # Circle fit to validate trajectory shape
            pts    = np.column_stack([xs, ys])
            circle = fit_circle_ransac(pts)
            if circle is None:
                continue

            cx_c, cy_c, r_c, _ = circle
            if arc_coverage(pts, cx_c, cy_c) < MIN_ARC:
                continue

            matches.append({'fid': fid, 'history': hist, 'circle': circle})

        return matches


def _znorm(arr: np.ndarray):
    """Z-normalise an array; return None if std ≈ 0."""
    s = float(np.std(arr))
    if s < 1e-6:
        return None
    return (arr - float(np.mean(arr))) / s


# ═════════════════════════════════════════════════════════════════════════════
#  ROI INITIALISER  (Phase 2 – dense Farnebäck OF + connected components)
# ═════════════════════════════════════════════════════════════════════════════

def initialise_roi(prev_g: np.ndarray, curr_g: np.ndarray,
                   histories: list, cam_w: int, cam_h: int):
    """
    Phase 2: Build the initial ROI bounding box from the dense optical flow.

    Steps (follow Eq. 1–7 in the paper):
      1. Compute trajectory vectors A = {a_i} from matched feature histories.
      2. Find average unit vector → average angle θ̄; find D_min, D_max.
      3. Compute dense Farnebäck OF.
      4. Pixel mask: magnitude in [(1−TH_D)·D_min, (1+TH_D)·D_max]    (Eq. 6)
                     angle within TH_ANGL of θ̄                         (Eq. 7)
      5. Seeded connected-component labelling from feature positions.
      6. Bounding box of the resulting component → initial ROI.
    """
    # Step 1 & 2 ─────────────────────────────────────────────────────────────
    traj_vecs = []
    seed_pts  = []
    for hist in histories:
        if len(hist) >= 2:
            # displacement over the last frame (Eq. 1: a_i = prev – curr)
            dx = hist[-2][0] - hist[-1][0]
            dy = hist[-2][1] - hist[-1][1]
            traj_vecs.append((dx, dy))
        if hist:
            seed_pts.append((int(hist[-1][0]), int(hist[-1][1])))

    if not traj_vecs or not seed_pts:
        return None

    arr  = np.array(traj_vecs, dtype=np.float64)
    mags = np.linalg.norm(arr, axis=1)
    D_min, D_max = float(mags.min()), float(mags.max())

    units   = arr / (mags[:, None] + 1e-9)
    avg_u   = units.mean(axis=0)                   # Eq. 2: ā
    avg_ang = float(np.arctan2(avg_u[1], avg_u[0])) # θ̄ = ∠ā

    # Step 3 ─────────────────────────────────────────────────────────────────
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flow_ang = np.arctan2(flow[..., 1], flow[..., 0])

    # Step 4 – pixel mask  (Eq. 6–7) ─────────────────────────────────────────
    lo = max(0.3, (1.0 - TH_D) * D_min)
    hi = (1.0 + TH_D) * max(D_max, 0.5)
    mask_m = (flow_mag >= lo) & (flow_mag <= hi)

    ang_diff = np.abs(flow_ang - avg_ang)
    ang_diff = np.minimum(ang_diff, 2.0 * math.pi - ang_diff)
    mask_a = ang_diff < TH_ANGL

    mask = (mask_m & mask_a).astype(np.uint8)

    # Step 5 – seeded connected-component labelling ────────────────────────
    seed_mask = np.zeros((cam_h, cam_w), dtype=np.uint8)
    for sx, sy in seed_pts:
        x1, x2 = max(0, sx - 5), min(cam_w, sx + 5)
        y1, y2 = max(0, sy - 5), min(cam_h, sy + 5)
        seed_mask[y1:y2, x1:x2] = 1

    _, labels, _, _ = cv2.connectedComponentsWithStats(mask)
    seed_labels = set(int(v) for v in labels[seed_mask == 1]) - {0}

    comp_mask = np.zeros_like(mask)
    for lv in seed_labels:
        comp_mask[labels == lv] = 1

    pts_in = np.argwhere(comp_mask == 1)

    # Step 6 – bounding box ────────────────────────────────────────────────
    if pts_in.size == 0:
        # Fallback: 30×30 px box around the feature seed (paper: "minimum ROI")
        sxs, sys_ = zip(*seed_pts)
        x1 = max(0, min(sxs) - 15)
        y1 = max(0, min(sys_) - 15)
        x2 = min(cam_w, max(sxs) + 15)
        y2 = min(cam_h, max(sys_) + 15)
        return x1, y1, max(MIN_ROI, x2 - x1), max(MIN_ROI, y2 - y1)

    ys_c, xs_c = pts_in[:, 0], pts_in[:, 1]
    x1 = max(0,     int(xs_c.min()) - 4)
    y1 = max(0,     int(ys_c.min()) - 4)
    x2 = min(cam_w, int(xs_c.max()) + 4)
    y2 = min(cam_h, int(ys_c.max()) + 4)
    return x1, y1, max(MIN_ROI, x2 - x1), max(MIN_ROI, y2 - y1)


# ═════════════════════════════════════════════════════════════════════════════
#  MODIFIED MEDIAN FLOW TRACKER  (Phase 3)
# ═════════════════════════════════════════════════════════════════════════════

def _caterer_grid(w: int, h: int) -> list:
    """
    Generate tracking points with spacing inspired by the central polygonal /
    lazy caterer's sequence (paper §"Tracking"), which denses points toward
    the centre to reduce the chance of latching onto background pixels.
    """
    step = max(6, int(math.sqrt(w * h / 25)))
    return [[x, y] for y in range(step, h, step) for x in range(step, w, step)]


class MedianFlowTracker:
    """
    Modified Median Flow tracker as described in the paper (§ "Tracking"):

      • Tracks an ROI using a grid of LK-tracked points (Median Flow).
      • Recalibrates the ROI periodically via dense OF to keep the bounding
        box tight around the body part / object.
      • Records a position offset on recalibration so the cursor does not jump
        (paper: "the recalibration is unnoticeable to the user").
      • Jitter filter: dynamic moving-window average  (Eq. 9).

    The cursor position is computed as:
      cursor = start_cursor + (roi_centre - initial_roi_centre) × cd_gain
    which implements a relative, gain-scaled absolute control map.
    """

    _LK = dict(winSize=(15, 15), maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    def __init__(self, roi: tuple, cd_gain: float,
                 prev_g: np.ndarray, cam_w: int, cam_h: int,
                 start_cursor: tuple):
        x, y, w, h = roi
        self.roi    = [float(x), float(y), float(w), float(h)]
        self.cd_gain      = cd_gain
        self.cam_w        = cam_w
        self.cam_h        = cam_h
        self.start_cursor = start_cursor       # (cx, cy) display px at coupling time
        self._roi0_cx     = x + w / 2.0       # ROI centre at coupling time (cam coords)
        self._roi0_cy     = y + h / 2.0

        self._off_x       = 0.0               # compensation offset (paper §"Tracking")
        self._off_y       = 0.0

        self._pos_hist    = deque(maxlen=JITTER_NMAX)
        self._pos_hist.append(start_cursor)
        self._last_pos    = start_cursor

        self._last_g      = prev_g
        self._last_recal  = time.time()
        self._last_moved  = time.time()        # used for idle detection

    # ── public API ────────────────────────────────────────────────────────

    def update(self, g: np.ndarray) -> tuple:
        """Advance one frame.  Returns (cursor_x, cursor_y) in display coords."""
        x, y, w, h = (int(v) for v in self.roi)

        # ── Median Flow step ─────────────────────────────────────────────
        grid = _caterer_grid(w, h)
        if not grid:
            self._last_g = g
            return self._last_pos

        old_pts = np.array([[[x + gx, y + gy]] for gx, gy in grid],
                           dtype=np.float32)
        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._last_g, g, old_pts, None, **self._LK)

        valid = st.ravel() == 1
        if valid.sum() < 4:
            self._last_g = g
            return self._last_pos

        disp    = new_pts[valid] - old_pts[valid]
        dx_med  = float(np.median(disp[:, 0, 0]))
        dy_med  = float(np.median(disp[:, 0, 1]))

        # move ROI
        nx = float(np.clip(self.roi[0] + dx_med, 0, self.cam_w - self.roi[2]))
        ny = float(np.clip(self.roi[1] + dy_med, 0, self.cam_h - self.roi[3]))
        move = math.hypot(nx - self.roi[0], ny - self.roi[1])
        self.roi[0], self.roi[1] = nx, ny

        if move > 0.3:
            self._last_moved = time.time()

        # ── periodic recalibration ────────────────────────────────────────
        t_now = time.time()
        if move > RECALIB_MOVE and (t_now - self._last_recal) > RECALIB_INT:
            self._recalibrate(g)

        # ── raw ROI centre (cam coords) with offset compensation ──────────
        raw_cx = self.roi[0] + self.roi[2] / 2.0 + self._off_x
        raw_cy = self.roi[1] + self.roi[3] / 2.0 + self._off_y

        # ── cursor via CD-gain  (Eq. 8 applied as relative offset) ───────
        cur_x = self.start_cursor[0] + (raw_cx - self._roi0_cx) * self.cd_gain
        cur_y = self.start_cursor[1] + (raw_cy - self._roi0_cy) * self.cd_gain
        cur_x = float(np.clip(cur_x, 0, WIN_W - 1))
        cur_y = float(np.clip(cur_y, 0, WIN_H - 1))

        # ── jitter filter  (Eq. 9) ────────────────────────────────────────
        if self._pos_hist:
            px, py = self._pos_hist[-1]
            dt = math.hypot(cur_x - px, cur_y - py)
            if dt < JITTER_DMIN:
                # NB = NMAX − ⌊dt × NMAX / dMIN⌋
                nb     = max(1, JITTER_NMAX - int(dt * JITTER_NMAX / JITTER_DMIN))
                recent = list(self._pos_hist)[-nb:] + [(cur_x, cur_y)]
                cur_x  = float(np.mean([p[0] for p in recent]))
                cur_y  = float(np.mean([p[1] for p in recent]))

        self._pos_hist.append((cur_x, cur_y))
        self._last_pos = (cur_x, cur_y)
        self._last_g   = g
        return self._last_pos

    def is_idle(self) -> bool:
        return (time.time() - self._last_moved) > IDLE_TIMEOUT

    # ── recalibration (dense OF re-fit) ──────────────────────────────────

    def _recalibrate(self, g: np.ndarray):
        """
        Re-run ROI detection using dense OF in a 2× neighbourhood.
        Records offset so the cursor remains stable (paper: 'unnoticeable').
        """
        cx_pre = self.roi[0] + self.roi[2] / 2.0
        cy_pre = self.roi[1] + self.roi[3] / 2.0

        rx, ry = int(self.roi[0]), int(self.roi[1])
        rw, rh = int(self.roi[2]), int(self.roi[3])
        sx = max(0, rx - rw // 2);  ex = min(self.cam_w, rx + rw + rw // 2)
        sy = max(0, ry - rh // 2);  ey = min(self.cam_h, ry + rh + rh // 2)

        reg_p = self._last_g[sy:ey, sx:ex]
        reg_c = g[sy:ey, sx:ex]
        if reg_p.shape[0] < 10 or reg_p.shape[1] < 10:
            self._last_recal = time.time()
            return

        flow = cv2.calcOpticalFlowFarneback(
            reg_p, reg_c, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_ang = np.arctan2(flow[..., 1], flow[..., 0])

        # Reference from last two cursor positions (mapped to cam scale)
        if len(self._pos_hist) >= 2:
            p0, p1 = self._pos_hist[-2], self._pos_hist[-1]
            roi_dx = (p1[0] - p0[0]) / WIN_W * self.cam_w
            roi_dy = (p1[1] - p0[1]) / WIN_H * self.cam_h
        else:
            self._last_recal = time.time()
            return

        roi_mag = math.hypot(roi_dx, roi_dy)
        if roi_mag < 0.1:
            self._last_recal = time.time()
            return
        roi_ang = math.atan2(roi_dy, roi_dx)

        lo = (1.0 - TH_D) * roi_mag * 0.5
        hi = (1.0 + TH_D) * roi_mag * 2.0
        mask_m = (flow_mag >= lo) & (flow_mag <= hi)
        ang_d  = np.minimum(np.abs(flow_ang - roi_ang),
                            2.0 * math.pi - np.abs(flow_ang - roi_ang))
        mask   = (mask_m & (ang_d < TH_ANGL)).astype(np.uint8)

        # Seed from current ROI centre in local region coords
        slx = int(np.clip(cx_pre - sx, 0, mask.shape[1] - 1))
        sly = int(np.clip(cy_pre - sy, 0, mask.shape[0] - 1))
        sm  = np.zeros_like(mask)
        sm[max(0, sly-5):sly+5, max(0, slx-5):slx+5] = 1

        _, labels, _, _ = cv2.connectedComponentsWithStats(mask)
        slabels = set(int(v) for v in labels[sm == 1]) - {0}

        if not slabels:
            self._last_recal = time.time()
            return

        comp = np.zeros_like(mask)
        for sl in slabels:
            comp[labels == sl] = 1

        pts_in = np.argwhere(comp == 1)
        if len(pts_in) < 4:
            self._last_recal = time.time()
            return

        ys_c, xs_c = pts_in[:, 0] + sy, pts_in[:, 1] + sx
        nx1 = max(0,         int(xs_c.min()) - 4)
        ny1 = max(0,         int(ys_c.min()) - 4)
        nx2 = min(self.cam_w, int(xs_c.max()) + 4)
        ny2 = min(self.cam_h, int(ys_c.max()) + 4)
        nw  = max(MIN_ROI, nx2 - nx1)
        nh  = max(MIN_ROI, ny2 - ny1)

        cx_post = nx1 + nw / 2.0
        cy_post = ny1 + nh / 2.0

        # Record offset to prevent cursor jump (paper §"Tracking")
        self._off_x += cx_pre - cx_post
        self._off_y += cy_pre - cy_post

        self.roi = [float(nx1), float(ny1), float(nw), float(nh)]
        self._last_recal = time.time()


# ═════════════════════════════════════════════════════════════════════════════
#  TARGET WIDGET  (dwell-to-activate demo target)
# ═════════════════════════════════════════════════════════════════════════════

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

    def draw(self, canvas: np.ndarray):
        if self.activated:
            cv2.circle(canvas, (self.cx, self.cy), self.r, C_TGT_DONE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (self.cx, self.cy), self.r, C_WHITE,    2,  cv2.LINE_AA)
        else:
            frac = 0.0
            if self.dwell_t0 is not None:
                frac = min(1.0, (time.time() - self.dwell_t0) / DWELL_TIME)
            col = C_TGT_DWELL if frac > 0 else C_TGT_IDLE
            cv2.circle(canvas, (self.cx, self.cy), self.r, col, 2, cv2.LINE_AA)
            if frac > 0:
                cv2.ellipse(canvas, (self.cx, self.cy),
                            (self.r + 8, self.r + 8), -90, 0, int(frac * 360),
                            C_TGT_DWELL, 5, cv2.LINE_AA)

        if self.label:
            fs   = 0.75
            size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
            tx   = self.cx - size[0] // 2
            ty   = self.cy + size[1] // 2
            cv2.putText(canvas, self.label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, C_WHITE, 2, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class MatchPointApp:
    """
    Application state machine:
      'matching' → orbit visible, motion matcher runs each frame
      'coupled'  → ROI tracker active, cursor visible, targets interactive
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam (device 0).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.orbit     = Orbit(WIN_W // 2, WIN_H // 2)
        self.targets   = self._make_targets()

        self.feat_tracker = FeatureTracker()
        self.mot_matcher  = MotionMatcher(self.orbit)
        self.roi_tracker  = None      # MedianFlowTracker (active when coupled)
        self.cursor       = None      # (x, y) display coords

        self.state   = 'matching'
        self.fps     = 30.0
        self._t_buf  = deque(maxlen=30)
        self._prev_g = None
        self._debug  = False

        cv2.namedWindow("MatchPoint", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MatchPoint", WIN_W, WIN_H)

    # ── target layout ─────────────────────────────────────────────────────

    def _make_targets(self) -> list:
        cx, cy = WIN_W // 2, WIN_H // 2
        R      = 290
        return [Target(int(cx + R * math.cos(i * math.pi / 2)),
                       int(cy + R * math.sin(i * math.pi / 2)),
                       r=45, label=lbl)
                for i, lbl in enumerate(["A", "B", "C", "D"])]

    # ── fps tracking ──────────────────────────────────────────────────────

    def _update_fps(self):
        t = time.time()
        self._t_buf.append(t)
        if len(self._t_buf) >= 2:
            self.fps = max(5.0, len(self._t_buf) /
                           (self._t_buf[-1] - self._t_buf[0] + 1e-9))
        self.mot_matcher.fps = self.fps

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        print("─" * 60)
        print("MatchPoint  |  UIST 2017 replication")
        print("Mirror the orbiting dot to acquire a touchless cursor.")
        print("Controls:  d = debug overlay   r = reset   q = quit")
        print("─" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self._update_fps()
            frame = cv2.flip(frame, 1)                  # mirror for natural use
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            canvas = self._build_canvas(frame)

            if self.state == 'matching':
                self._run_matching(gray, canvas)
            else:
                self._run_coupled(gray, canvas)

            self._draw_hud(canvas)
            cv2.imshow("MatchPoint", canvas)
            self._prev_g = gray

            k = cv2.waitKey(1) & 0xFF
            if   k == ord('q'):  break
            elif k == ord('d'):  self._debug = not self._debug
            elif k == ord('r'):  self._reset()

        self.cap.release()
        cv2.destroyAllWindows()

    # ── canvas ────────────────────────────────────────────────────────────

    def _build_canvas(self, frame: np.ndarray) -> np.ndarray:
        """Dim the webcam feed and use it as the background layer."""
        bg   = cv2.resize(frame, (WIN_W, WIN_H))
        dark = np.zeros_like(bg)
        return cv2.addWeighted(bg, 0.22, dark, 0.78, 0)

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def _run_matching(self, gray: np.ndarray, canvas: np.ndarray):
        self.feat_tracker.update(gray)
        self.orbit.draw(canvas)
        self._put_hint(canvas,
            "Mirror the orbiting dot with any body movement to acquire a pointer")

        if self._debug and self.feat_tracker._pts is not None:
            for pt in self.feat_tracker._pts:
                cv2.circle(canvas,
                           (int(pt[0][0] * SCALE_X), int(pt[0][1] * SCALE_Y)),
                           2, C_FEAT_DBG, -1)

        if self._prev_g is None:
            return

        t_now   = time.time()
        matches = self.mot_matcher.check(self.feat_tracker.full_tracks(), t_now)
        if not matches:
            return

        m = matches[0]   # take first (all are valid; one is sufficient)
        hist   = m['history']
        circle = m['circle']
        _, _, r_fit, _ = circle

        roi = initialise_roi(self._prev_g, gray, [hist], CAM_W, CAM_H)
        if roi is None:
            return

        # CD-gain  (Eq. 8):  CDgain = (1/r) × screen_dim
        cd_gain = (1.0 / max(r_fit, 4.0)) * max(WIN_W, WIN_H)

        # Cursor starts at the mapped position of the matched feature
        roi_cx = roi[0] + roi[2] / 2.0
        roi_cy = roi[1] + roi[3] / 2.0
        start  = (float(np.clip(roi_cx * SCALE_X, 0, WIN_W - 1)),
                  float(np.clip(roi_cy * SCALE_Y, 0, WIN_H - 1)))

        self.roi_tracker = MedianFlowTracker(
            roi, cd_gain, gray, CAM_W, CAM_H, start)
        self.cursor = (int(start[0]), int(start[1]))
        self.state  = 'coupled'
        print(f"[coupled]  roi={roi}  r={r_fit:.1f}px  cd_gain={cd_gain:.2f}")

    # ── Phase 3 ───────────────────────────────────────────────────────────

    def _run_coupled(self, gray: np.ndarray, canvas: np.ndarray):
        if self.roi_tracker is None:
            self._reset()
            return

        cx, cy     = self.roi_tracker.update(gray)
        self.cursor = (int(cx), int(cy))

        for tgt in self.targets:
            tgt.update(self.cursor)
            tgt.draw(canvas)

        # Cursor crosshair
        cv2.circle(canvas, self.cursor, 16, C_CURSOR,    -1, cv2.LINE_AA)
        cv2.circle(canvas, self.cursor, 20, C_CURSOR_RIM,  2, cv2.LINE_AA)

        if self._debug:
            rx, ry, rw, rh = self.roi_tracker.roi
            p1 = (int(rx * SCALE_X), int(ry * SCALE_Y))
            p2 = (int((rx + rw) * SCALE_X), int((ry + rh) * SCALE_Y))
            cv2.rectangle(canvas, p1, p2, C_ROI_DBG, 1)

        self._put_hint(canvas,
            "Move to control cursor  |  Hover a target to select  |  Still = release")

        if self.roi_tracker.is_idle():
            print("[decoupled]  idle timeout")
            self._reset()

    # ── HUD ──────────────────────────────────────────────────────────────

    def _draw_hud(self, canvas: np.ndarray):
        label = "MATCHING" if self.state == 'matching' else "COUPLED "
        txt   = f"MatchPoint  |  {label}  |  {self.fps:.0f} fps"
        cv2.putText(canvas, txt, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_GREY, 2, cv2.LINE_AA)
        if self._debug:
            cv2.putText(canvas, "DEBUG", (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ORBIT_DOT, 1, cv2.LINE_AA)

    @staticmethod
    def _put_hint(canvas: np.ndarray, text: str):
        fs = 0.55
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
        cv2.putText(canvas, text, ((WIN_W - tw) // 2, WIN_H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, C_DIM, 1, cv2.LINE_AA)

    # ── reset ─────────────────────────────────────────────────────────────

    def _reset(self):
        self.state        = 'matching'
        self.roi_tracker  = None
        self.cursor       = None
        self.feat_tracker = FeatureTracker()
        for t in self.targets:
            t.dwell_t0  = None
            t.activated = False
        print("[reset]  back to matching state")


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MatchPointApp().run()
