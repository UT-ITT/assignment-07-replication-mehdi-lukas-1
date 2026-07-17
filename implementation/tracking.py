import cv2
import math
import time
import numpy as np
from collections import deque

# Import globals
from globals import TH_D, TH_ANGL, MIN_ROI
from globals import RECALIB_MOVE, RECALIB_INT
from globals import JITTER_DMIN, JITTER_NMAX

from globals import WIN_W, WIN_H
from globals import IDLE_TIMEOUT, DWELL_TIME

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
        if len(hist) > 0:
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

    def update(self, g: np.ndarray, WIN_W, WIN_H) -> tuple:
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

        if move > 1:
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

