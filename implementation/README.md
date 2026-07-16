# MatchPoint — Touchless Pointing via Spontaneous Spatial Coupling

> **Replication of:**  
> Clarke, C. & Gellersen, H. (2017). *MatchPoint: Spontaneous Spatial Coupling of Body Movement for Touchless Pointing.*  
> UIST '17 — Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology, pp. 179–192.  
> DOI: [10.1145/3126594.3126626](https://doi.org/10.1145/3126594.3126626) · [Full paper (PDF)](https://eprints.lancs.ac.uk/id/eprint/88665/1/MatchPoint_author_version.pdf)

---

## Table of Contents

1. [What Is MatchPoint?](#1-what-is-matchpoint)
2. [How the Interaction Works — Step by Step](#2-how-the-interaction-works--step-by-step)
3. [The Algorithm — Under the Hood](#3-the-algorithm--under-the-hood)
   - [Phase 1 — Motion Matching](#phase-1--motion-matching)
   - [Phase 2 — ROI Initialisation](#phase-2--roi-initialisation)
   - [Phase 3 — Modified Median Flow Tracking](#phase-3--modified-median-flow-tracking)
   - [Pointer Termination](#pointer-termination)
4. [System Requirements](#4-system-requirements)
5. [Installation](#5-installation)
6. [Running the Application](#6-running-the-application)
7. [Keyboard Controls](#7-keyboard-controls)
8. [What You Will See on Screen](#8-what-you-will-see-on-screen)
9. [How to Successfully Acquire a Cursor](#9-how-to-successfully-acquire-a-cursor)
10. [Tunable Parameters](#10-tunable-parameters)
11. [Code Structure](#11-code-structure)
12. [Design Decisions & Deviations from the Paper](#12-design-decisions--deviations-from-the-paper)
13. [Known Limitations](#13-known-limitations)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What Is MatchPoint?

**MatchPoint** is a **touchless pointing system** — a way to control a cursor on screen using nothing but a standard webcam and any movement you can produce with your body (hand, head, elbow, foot, or even an object you hold). No special hardware, no gloves, no markers, no depth sensors, no machine learning training data needed.

The core idea is called **Spontaneous Spatial Coupling**:

> The screen shows a widget (called an **Orbit**) — a ring with a small dot that travels around its circumference at a fixed speed. When the user **mirrors** that circular motion with any body part, the system detects the synchrony and **couples** the body part to a cursor. From that moment the cursor follows every movement of the matched body part, scaled appropriately to fill the screen.

The "spontaneous" part is key: the user does not need to register, calibrate, or hold a special pose. They simply **copy the dot's motion** and a pointer appears.

This prototype replicates the single-pointer, single-orbit core of the paper — the interaction technique itself — without the multi-user or TV-remote application layers described in the publication.

---

## 2. How the Interaction Works — Step by Step

```
┌─────────────────────────────────────────────────────────────┐
│  STATE: MATCHING                                            │
│                                                             │
│  Screen shows: ○ (ring) with a dot ● orbiting it           │
│  Camera watches: all motion in the field of view            │
│                                                             │
│  User action:  move your hand/head in a circle that         │
│                follows the dot's rhythm                     │
│                                                             │
│  System detects: your motion correlates with the orbit      │
│                  signal → MATCH confirmed                    │
└──────────────────────────┬──────────────────────────────────┘
                           │  coupling event
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STATE: COUPLED                                             │
│                                                             │
│  Cursor ✜ appears on screen, anchored to your body part    │
│  Move your hand/head → cursor follows, amplified by gain   │
│                                                             │
│  Demo targets A B C D are shown                             │
│  Hover cursor over a target for 0.8 s → it activates (✓)  │
│                                                             │
│  Stay completely still for 3 s → cursor decouples          │
│  Press R → manually reset to MATCHING                       │
└─────────────────────────────────────────────────────────────┘
```

The user never clicks a button to "start". The very act of mimicking the orbit's motion *is* the activation signal — making the technique truly spontaneous.

---

## 3. The Algorithm — Under the Hood

The implementation is a faithful replication of the three-pipeline architecture described in the paper.

---

### Phase 1 — Motion Matching

**Goal:** Detect when the user's movement correlates with the Orbit.

#### 3.1.1 Orbit Widget

The Orbit is centred at the screen midpoint. Its orbiting dot moves at a fixed angular velocity:

```
angle(t) = (2π / T) · t      where T = 2.5 s (one full revolution)
dot_x(t) = cx + R · cos(angle(t))
dot_y(t) = cy + R · sin(angle(t))
```

The x-signal is a **cosine wave** and the y-signal is a **sine wave**, both with period 2.5 s. These are the reference signals used for correlation.

#### 3.1.2 FAST Feature Detection

Each webcam frame is converted to greyscale. The **FAST** (Features from Accelerated Segment Test) detector finds corner-like keypoints — image regions that move distinctively and can be reliably tracked. Settings: threshold 20, non-maximum suppression ON. Points are detected on first run and re-detected every 15 frames (or when fewer than 60 remain).

#### 3.1.3 Lucas-Kanade Optical Flow

Each detected FAST keypoint is propagated frame-to-frame using **pyramidal Lucas-Kanade** optical flow (window 21×21, 3 pyramid levels). This gives each feature a trajectory — a list of (x, y) positions over the last `HIST_LEN = 38` frames (≈ 1.25 seconds at 30 fps, covering half a revolution of the orbit).

#### 3.1.4 Pearson Correlation Check

For each feature with a full 38-frame trajectory:

1. Retrieve the Orbit's expected x-trajectory and y-trajectory for the same 38-frame window.
2. **Z-normalise** both signals (subtract mean, divide by standard deviation). This makes the comparison scale-invariant — a hand close to the camera moving in a large circle and a head far away moving in a small circle both yield the same correlation score if their *rhythm* matches.
3. Compute `rx = Pearson(feature_x_normalised, orbit_x_normalised)` and `ry = Pearson(feature_y_normalised, orbit_y_normalised)`.
4. If `max(|rx|, |ry|) ≥ 0.75` → potential match.

The threshold of 0.75 is taken directly from the paper and ensures accidental matches are rare.

#### 3.1.5 RANSAC Circle Fit + Arc Coverage

Correlation alone does not prove circular motion (a straight-line oscillation at the right frequency would also correlate). So the trajectory is additionally validated by fitting a circle using **RANSAC**:

- **120 iterations**, each picking 3 random trajectory points and solving for the unique circumscribed circle.
- Inlier threshold: 8 px from the fitted radius.
- Accept if ≥ 6 inliers.
- **Arc coverage**: discretise the inlier points' angles into 16 bins (22.5° each); require at least 4 non-empty bins (≥ 25% of the circle covered). This confirms the user traced an arc, not a straight segment.

Only if both correlation **and** circle fit pass is the feature declared a **match**.

---

### Phase 2 — ROI Initialisation

**Goal:** Find the spatial extent of the matched body part / object in the camera frame, and compute the cursor gain.

This phase runs once per coupling event.

#### 3.2.1 Displacement Vectors

For each matched feature point, compute the displacement between the last two frames:

```
a_i = (x_{t-1} - x_t,  y_{t-1} - y_t)        (Eq. 1 in paper)
```

From all matched features, compute:
- Magnitude range: D_min, D_max
- Average direction angle: `θ̄ = ∠( mean(â_i) )`   (average unit vector)   (Eq. 2)

#### 3.2.2 Dense Optical Flow

The **Farnebäck dense optical flow** algorithm computes a motion vector for **every pixel** in the frame (not just keypoints). This gives a full picture of what is moving where.

#### 3.2.3 Pixel Mask (Equations 6 & 7 from the paper)

Keep only pixels where:

```
(1 − 0.25) · D_min  ≤  |flow_i|  ≤  (1 + 0.25) · D_max     [Eq. 6]
|∠flow_i − θ̄|  <  π/8    (≈ 22.5°)                          [Eq. 7]
```

This isolates pixels moving with the same **speed and direction** as the matched feature — which should be the body part / object that triggered the match.

#### 3.2.4 Connected-Component Labelling

**Connected-component labelling** groups the masked pixels into blobs. The algorithm is **seeded** from a 10×10 px area around each matched feature point. Only blobs connected to these seeds are retained — this avoids picking up other objects in the background that happen to have similar motion.

The bounding box of the resulting blob becomes the **initial ROI** (Region of Interest).

Fallback: if no blob is found, a 30×30 px minimum box around the feature point is used (paper §"Tracker Initialisation").

#### 3.2.5 CD-Gain Calculation (Equation 8 from the paper)

The Control-Display (CD) gain maps camera-space movement to screen-space cursor movement:

```
CD_gain = (1 / r_fit) × max(screen_width, screen_height)
```

Where `r_fit` is the radius of the RANSAC-fitted circle — a proxy for how large the user's matching motion was. If the user made a small head nod (r_fit ≈ 15 px), the gain is high (cursor travels far per unit movement). If the user swung their arm in a large arc (r_fit ≈ 80 px), the gain is lower. This self-calibrates the pointer to the user's natural movement range.

---

### Phase 3 — Modified Median Flow Tracking

**Goal:** Track the ROI frame-to-frame and translate its motion into cursor movement.

#### 3.3.1 Grid of Tracking Points

Instead of tracking a single point, a **grid of points** is placed inside the ROI. Spacing is derived from `sqrt(ROI_area / 25)`, keeping ~25 points distributed across the ROI interior. This is inspired by the "central polygonal / lazy caterer's sequence" spacing mentioned in the paper, which avoids concentrating points near the edges where background pixels may creep in.

#### 3.3.2 Median Flow

Each frame:
1. Track all grid points with LK optical flow.
2. Keep only successfully tracked points (LK status = 1).
3. Compute the **median** x-displacement and median y-displacement across all valid points.
4. Move the ROI by this median vector.

The median (instead of mean) is robust to outliers — background pixels that accidentally moved differently will not drag the ROI.

#### 3.3.3 Periodic Recalibration

Over time the ROI can drift or the body part may change shape/orientation. Every **500 ms** (when ROI moved > 2 px), the system:
1. Runs dense Farnebäck OF in a **2× expanded search region** around the current ROI.
2. Re-applies the pixel mask (Eq. 6–7) using the ROI's recent displacement as the reference.
3. Re-runs connected-component labelling seeded from the current ROI centre.
4. Fits a new bounding box.
5. Records the **offset** between the old and new ROI centre so the cursor does **not jump** — the paper explicitly states *"the recalibration is unnoticeable to the user"*.

#### 3.3.4 Jitter Filter (Equation 9 from the paper)

Camera noise causes the tracker to produce small random displacements even when the user is still. The jitter filter suppresses this with a **dynamic moving-window average**:

```
NB = NMAX − ⌊ dt × NMAX / dMIN ⌋
```

Where:
- `dt` = Euclidean distance from previous cursor position (px)
- `dMIN = 2.5 px` — below this, jitter filtering is active
- `NMAX = 10` — maximum window size (frames)

When `dt` is very small (still), `NB` is large → heavy smoothing.  
When `dt` is large (moving fast), `NB` → 1 → no smoothing, full responsiveness.

#### 3.3.5 Cursor Mapping

```
cursor_x = start_x + (roi_cx_now − roi_cx_initial) × CD_gain
cursor_y = start_y + (roi_cy_now − roi_cy_initial) × CD_gain
```

The cursor starts where the matched body part maps to on screen and moves with amplified displacement from there.

---

### Pointer Termination

If the tracked ROI centre moves less than 0.3 px for **3 continuous seconds**, the system interprets this as the user having finished interacting. The cursor decouples and the system returns to the MATCHING state. The orbit reappears and the user can acquire a new pointer at will.

---

## 4. System Requirements

| Component | Minimum | Tested |
|---|---|---|
| **Operating System** | macOS 11 / Linux / Windows 10 | macOS 15 (Apple Silicon) |
| **Python** | 3.10 | 3.14 |
| **Webcam** | Any USB or built-in RGB camera | MacBook built-in FaceTime HD |
| **OpenCV** | 4.5 | 5.0.0 |
| **NumPy** | 1.20 | 2.5.0 |
| **SciPy** | 1.7 | 1.18.0 |
| **RAM** | 512 MB | — |
| **CPU** | Any modern dual-core | Apple M2 |
| **GPU** | Not required | — |

> **No GPU, no special camera, no depth sensor, no training data.** A standard laptop webcam is all that is needed.

---

## 5. Installation

### Step 1 — Clone or download the repository

```bash
git clone <repository-url>
cd assignment-07-replication-mehdi-lukas-1
```

### Step 2 — (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r implementation/requirements.txt
```

Dependencies installed:

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture, FAST detection, Lucas-Kanade OF, Farnebäck dense OF, connected components, drawing |
| `numpy` | Array operations, trajectory vectors, masking |
| `scipy` | `pearsonr` — Pearson correlation coefficient for motion matching |

### Step 4 — Verify installation

```bash
python3 -c "import cv2, numpy, scipy; print('OK', cv2.__version__)"
```

Expected output: `OK 5.x.x` (or similar).

---

## 6. Running the Application

```bash
# From the repository root:
python3 implementation/matchpoint.py

# Or from inside the implementation folder:
cd implementation
python3 matchpoint.py
```

A window titled **"MatchPoint"** opens at **1280 × 720 px**. The webcam (device 0) starts automatically.

> **On macOS:** the system will request webcam permission the first time. Click *Allow* in the dialog that appears.

> **On Linux:** if device 0 fails, try setting `cv2.VideoCapture(1)` in the `MatchPointApp.__init__` method.

---

## 7. Keyboard Controls

| Key | Action |
|---|---|
| `d` | Toggle **debug overlay**: shows all tracked FAST feature points (green dots) and the current ROI bounding box (red rectangle) |
| `r` | **Reset** — immediately decouple and return to the MATCHING state; clears all target activations |
| `q` | **Quit** — close the window and release the webcam |

---

## 8. What You Will See on Screen

### MATCHING state

```
┌──────────────────────────────────────────────────┐
│  MatchPoint  |  MATCHING  |  30 fps              │
│                                                  │
│                    ╭────╮                        │
│                   /      \                       │
│                  │   ·    │                      │  ← Orbit ring (green)
│                   \  ●  /                        │  ← Orbiting dot (bright green)
│                    ╰────╯                        │
│                                                  │
│   Mirror the orbiting dot to acquire a pointer   │  ← hint text
└──────────────────────────────────────────────────┘
```

The dimmed webcam feed is visible in the background so you can see yourself. The bright green orbit is in the centre of the window.

### COUPLED state

```
┌──────────────────────────────────────────────────┐
│  MatchPoint  |  COUPLED   |  30 fps              │
│                                                  │
│           ● B                                    │
│                                                  │
│     ● A          ✜          ● C                  │  ← cursor (cyan)
│                                                  │
│           ● D                                    │
│                                                  │
│  Move to control cursor | Hover = select | r=reset│
└──────────────────────────────────────────────────┘
```

- **Cursor ✜** — cyan filled circle with rim. Follows your tracked body part.
- **Targets A B C D** — four blue circles arranged in a cross pattern.
  - **Hovering** over a target for 0.8 s: a progress arc draws around it in cyan.
  - **Activated** target: fills solid green (✓). Stays green until you press `r`.

---

## 9. How to Successfully Acquire a Cursor

The most important thing to understand: you are **not making a gesture**. You are **synchronising** with the orbit's circular motion.

### Step-by-step guide

1. **Open the app and look at the orbit dot.** Notice it travels in a circle every ~2.5 seconds.

2. **Raise your hand** so it is clearly visible to the webcam. Make sure you are in the webcam's field of view (your picture is dimly visible in the background of the window).

3. **Start moving your hand in a circle** — the same direction as the dot, at roughly the same speed. About the size of a tennis ball in the air is comfortable. You do not need to perfectly match the dot's exact position; you just need your motion to have the same **circular rhythm**.

4. **Keep going for about 1–1.5 seconds.** The system accumulates 38 frames of trajectory before it can confirm a match.

5. **The cursor appears** — a cyan dot materialises on screen where your hand is visible. You can now move your hand (more slowly and deliberately) to steer the cursor.

6. **To decouple:** simply hold your hand still for 3 seconds. The cursor disappears and the orbit reappears. You can then re-acquire with any body part.

### Tips

| ✅ Do | ❌ Avoid |
|---|---|
| Move smoothly and continuously in a circle | Stopping mid-circle or reversing direction |
| Match the orbit's **period** (~2.5 s/rev) | Moving much faster or slower than the dot |
| Start the motion while watching the dot | Starting before the app is fully loaded |
| Try with your **head** too (gentle nodding in a circle) | Keeping very still — the system needs motion to track |
| Use the **debug view** (`d`) to see tracked features | Wearing all-black clothing in a dark room |

### Why does it sometimes take a few seconds?

The Pearson correlation window is 38 frames. If your motion started mid-revolution, the system needs to observe at least half a full circle before the correlation peaks above 0.75. This is by design — it prevents accidental matches.

---

## 10. Tunable Parameters

All constants are at the top of `matchpoint.py` in clearly labelled sections. You can tune them without understanding the rest of the code.

### Display

| Constant | Default | Effect |
|---|---|---|
| `WIN_W, WIN_H` | `1280, 720` | Window size in pixels |
| `CAM_W, CAM_H` | `640, 480` | Webcam capture resolution |

### Orbit

| Constant | Default | Effect |
|---|---|---|
| `ORBIT_R` | `55` px | Radius of the orbit ring |
| `ORBIT_DOT_R` | `13` px | Size of the orbiting dot |
| `ORBIT_T` | `2.5` s | Period of one revolution — **increase** to make it slower and easier to match |

### Phase 1 — Motion Matching

| Constant | Default | Effect |
|---|---|---|
| `HIST_LEN` | `38` frames | History window. Must be > half a period × fps. Longer = more reliable but slower to trigger. |
| `MIN_DISP_PX` | `2.0` px | Minimum total feature displacement. Raise to ignore tiny fidgets. |
| `PEARSON_THRESH` | `0.75` | Correlation threshold. **Lower** (e.g. 0.65) → triggers more easily but with more false positives. **Raise** (e.g. 0.85) → requires closer matching. |
| `RANSAC_N_ITER` | `120` | More iterations = more robust circle fit but slower. |
| `RANSAC_EPS` | `8.0` px | Inlier tolerance for circle fit. Larger = accepts noisier trajectories. |
| `MIN_ARC` | `0.22` | Minimum arc fraction (0–1). Raise to require a more complete circle trace. |

### Phase 3 — Tracking

| Constant | Default | Effect |
|---|---|---|
| `JITTER_DMIN` | `2.5` px | Below this cursor movement, jitter filter activates. Raise if cursor is jittery when still. |
| `JITTER_NMAX` | `10` frames | Maximum smoothing window. Raise for smoother cursor at cost of input lag. |
| `IDLE_TIMEOUT` | `3.0` s | Seconds of stillness before decoupling. Lower for quicker release. |
| `DWELL_TIME` | `0.8` s | Seconds to hover over a target to activate it. |

---

## 11. Code Structure

```
implementation/
├── matchpoint.py           ← single self-contained application (999 lines)
└── requirements.txt        ← pip dependencies
```

### Inside `matchpoint.py`

```
Constants (lines 63–112)
│
├── class Orbit                   Generates the ring widget and reference signals
│
├── fit_circle_ransac()           RANSAC circle fit utility
├── arc_coverage()                Arc fraction estimator (16-bin discretisation)
├── _znorm()                      Z-normalisation helper
│
├── class FeatureTracker          FAST detection + LK optical flow + trajectory history
│
├── class MotionMatcher           Phase 1: Pearson + RANSAC → match detection
│
├── initialise_roi()              Phase 2: dense OF + connected components → ROI + CD-gain
│
├── _caterer_grid()               Grid-point generator for Median Flow
├── class MedianFlowTracker       Phase 3: Median Flow + recalibration + jitter filter
│
├── class Target                  Dwell-to-activate target widget
│
└── class MatchPointApp           Main application loop + state machine
```

Each class and function is documented with a docstring that references the specific equation or section of the paper it implements.

---

## 12. Design Decisions & Deviations from the Paper

| Decision | Rationale |
|---|---|
| **Single orbit, single pointer** | The paper describes multi-orbit / multi-pointer / tangible interfaces. We replicate the core coupling mechanism only, which is the academic contribution. |
| **Camera frame mirrored horizontally** | Moving your right hand → cursor moves right (natural mirror behaviour). Without mirroring, movement feels inverted. |
| **`HIST_LEN = 38` at 30 fps** | Half a revolution of a 2.5 s orbit ≈ 1.25 s × 30 fps = 37.5 → rounded to 38. Shorter windows risk spurious correlation; longer windows delay response. |
| **Cursor starts at ROI position** | The cursor appears at the display position corresponding to where the body part was when the match was detected. This avoids a large initial jump compared to starting at screen centre. |
| **Pearson correlation (scale-invariant)** | The paper uses Pearson r. This is crucial because camera-space amplitude has no fixed relationship to display-space amplitude; only the *shape* of the motion matters. |
| **16-bin arc coverage instead of continuous arc** | Computationally cheap; 16 bins (22.5° resolution) is more than sufficient to distinguish a circle from a line segment. |
| **Dense OF in 2× expanded region for recalibration** | Paper specifies searching in `2×W × 2×H` around the current ROI (§ "Tracking"). This is implemented verbatim. |
| **Farnebäck parameters unchanged from OpenCV defaults** | The paper cites the Farnebäck algorithm but does not specify parameters. OpenCV defaults (`pyr_scale=0.5, levels=3, winsize=15, iterations=3`) work well in practice. |

---

## 13. Known Limitations

| Limitation | Cause | Possible Fix |
|---|---|---|
| Slow matching in low-texture environments | FAST detector finds few reliable points on plain walls or uniform clothing | Move to a richer background or wear patterned clothing |
| False match if background contains periodic motion (fan, rotating object) | The Pearson check cannot distinguish intentional from coincidental motion | RANSAC circle fit partially mitigates this; avoid having rotating objects in frame |
| Tracking drift over long sessions | Modified Median Flow can accumulate small errors | The periodic recalibration limits drift; pressing `r` resets completely |
| CD-gain can be too high with very small matching motions | A tiny matching circle → small r_fit → very large gain → cursor overshoots | Move further from the camera or make a larger matching circle |
| Phase-shift sensitivity | If the user's circular motion lags the orbit by > ~60°, `|Pearson r|` drops below 0.75 | The user must start their circular motion while watching the dot and stay in sync |

---

## 14. Troubleshooting

**The window opens but the webcam shows a black frame**

The camera index may be wrong. Open `matchpoint.py` and change:
```python
self.cap = cv2.VideoCapture(0)   →   cv2.VideoCapture(1)  # or 2, etc.
```

**The cursor never appears even though I am making circular motions**

- Enable the debug view (`d`) and check that green feature points appear on your hand/body. If none appear, your background is likely too uniform — try wearing a patterned top or moving against a more textured wall.
- Try a **slower, larger** circular motion — match the dot's 2.5-second rhythm deliberately.
- Lower `PEARSON_THRESH` to `0.68` in the constants section.

**The cursor appears but is very jittery**

Raise `JITTER_DMIN` from `2.5` to `4.0` and `JITTER_NMAX` from `10` to `15`.

**The cursor moves way too fast / overshoots everything**

You made a very small matching circle → large CD-gain. Make a **bigger** circular motion when acquiring (at least fist-sized). Alternatively, cap the gain by adding `cd_gain = min(cd_gain, 40.0)` in `_run_matching`.

**The cursor barely moves**

You made an extremely large matching circle → small CD-gain. Make a **smaller** circular motion, or add a minimum: `cd_gain = max(cd_gain, 8.0)`.

**`ModuleNotFoundError: No module named 'cv2'`**

```bash
pip3 install opencv-python --break-system-packages
```

**macOS camera permission denied**

Go to *System Settings → Privacy & Security → Camera* and enable access for Terminal (or your IDE).

---

## References

- Clarke, C. & Gellersen, H. (2017). **MatchPoint: Spontaneous Spatial Coupling of Body Movement for Touchless Pointing.** In *Proceedings of UIST '17*, pp. 179–192. ACM. https://doi.org/10.1145/3126594.3126626
- Clarke, C., Bellino, A., Esteves, A., & Gellersen, H. (2016). **TraceMatch: A Computer Vision Method for User Interface Matching by Tracing of Animation Replays.** UIST '16.
- Esteves, A., Velloso, E., Bulling, A., & Gellersen, H. (2015). **Orbits: Gaze Interaction for Smart Watches using Smooth Pursuit Eye Movements.** UIST '15.
- Rosten, E. & Drummond, T. (2006). **Machine Learning for High-Speed Corner Detection.** ECCV '06. *(FAST detector)*
- Lucas, B. D. & Kanade, T. (1981). **An Iterative Image Registration Technique with an Application to Stereo Vision.** IJCAI '81. *(LK optical flow)*
- Farnebäck, G. (2003). **Two-Frame Motion Estimation Based on Polynomial Expansion.** SCIA '03. *(Dense optical flow)*
- Kalal, Z., Mikolajczyk, K., & Matas, J. (2010). **Forward-Backward Error: Automatic Detection of Tracking Failures.** ICPR '10. *(Median Flow)*

---

*Assignment 7 — Replicating Interaction Techniques · Interaction Techniques and Technologies 2026*  
*Authors: Mehdi Hoseyni & Lukas · Lancaster University MatchPoint Paper · UIST 2017*