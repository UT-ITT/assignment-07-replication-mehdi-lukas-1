# Interaction Techniques and Technologies 2026
## Assignment 7: Replicating Interaction Techniques

This document details our project's replication process, starting with the paper selection phase (Task 1).

---

### Task 1: Paper Selection

For this replication assignment, we discussed several human-computer interaction (HCI) papers, evaluating their feasibility, hardware requirements, implementation complexity, and overall suitability for a two-week development timeline.

#### Considered Papers & Feasibility Analysis

Below is the comparison table of all the papers considered by our team during the brainstorming phase:

| Paper Title & Citation | Interaction Modality | Hardware Required | Feasibility & Scope Evaluation | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **MatchPoint: Spontaneous Spatial Coupling of Body Movement for Touchless Pointing**<br>[Clarke & Gellersen, UIST 2017](https://eprints.lancs.ac.uk/id/eprint/88665/) | Touchless pointer control via spatial coupling of body parts with orbiting target widgets | Standard Web Camera / RGB Video Stream | **High**: Only requires a standard webcam. The tracking and correlation matching logic is self-contained and highly achievable within two weeks without training datasets. | **Selected** |
| **KeyFlow: Acoustic Motion Sensing for Cursor Control on Any Keyboard**<br>[Liu et al., UIST 2024](https://doi.org/10.1145/3654777.3676452) | Finger-gliding gestures on a keyboard surface tracked via keyboard acoustic vibration signals | Built-in or standard microphone | **Medium-Low**: Acoustic calibration is highly sensitive to background noise and varies significantly by keyboard model. Gathering training data for gesture classification in two weeks introduces high risk. | Excluded |
| **Mobile Phones as Pointing Devices**<br>[General HCI Literature / Remote Pointing](https://doi.org/10.1145/1520340.1520448) | Phone screen acts as virtual trackpad / camera optical flow mapping for large displays | Smartphone & computer screen | **Medium**: Concept is well-tested but slightly generic (similar to consumer remote mouse apps). The team preferred a more novel, research-focused gesture interaction technique. | Excluded |
| **Natural Throw and Tilt Interaction between Mobile Phones and Distant Displays**<br>[Dachselt & Buchholz, CHI 2009](https://doi.org/10.1145/1520340.1520448) | Directing content to/from displays via throw and tilt phone gestures | Accelerometer-equipped smartphone & large screen system | **Medium**: Requires multi-device sync (smartphone + receiver display) and network socket programming. Infrastructure overhead reduces time available for interaction polishing. | Excluded |
| **Mapping Gestures Performed with Tangible Cubes to System Commands**<br>[Latreche & Schiettecatte, 2026](https://uclouvain.be) | Rotating, tilting, or moving physical cubes to map to system commands | Custom physical cubes, sensor modules (RFID/IMU) or multi-camera tracking | **Low**: Requires specialized physical tangibles and/or complex multi-camera tracking systems to reliably distinguish 3D cube states, which is not feasible in a two-week window. | Excluded |
| **GazeBreath: Input Method Using Gaze Pointing and Breath Selection**<br>[Onishi et al., AH 2022](https://doi.org/10.1145/3519391.3519398) | Targeting screen objects via gaze and selecting/confirming via breath | Eye-tracker & Thermal Camera or breath-sensing microphone | **Low**: High barrier due to specialized hardware requirements. Gaze trackers and thermal cameras are not readily accessible to the team. | Excluded |
| **Immersive Emotions: Translating Emotions Into Visualization**<br>[Koo et al., MobileHCI 2022](https://doi.org/10.1145/3543174.3546083) | Mapping physiological signals (e.g., heart rate) to generative visualizations | Biosensors (Heart-rate / galvanic skin response sensors, sensor gloves) | **Low**: Relies on specific wearable biosensors that the team does not have access to, making replication impossible. | Excluded |
| **Augmented Chironomia for Presenting Data to Remote Audiences**<br>[Hall et al., UIST 2022](https://dl.acm.org/doi/10.1145/3526113.3545614) | Bimanual hand gesture tracking for presenting data to remote audiences | RGB-D/Depth Camera or advanced hand-tracking setup | **Medium-Low**: Focuses on complex gesture vocabulary mapped to remote data presentation tools. Replicating the full framework from scratch requires too much UI and data integration for two weeks. | Excluded |
| **Demonstration of GestuRING, a Web Tool for Ring Gesture Input**<br>[Bilius & Vatavu, UIST 2021](https://dblp.org/rec/conf/uist/BiliusV21) | Ring-shaped wearable gesture input for micro-interactions | Smart Ring or specialized micro-gesture wearable | **Low**: Requires custom wearable smart rings to capture micro-gestures, which are not available. | Excluded |

---

#### Justification for Selecting MatchPoint

We chose to replicate **"MatchPoint: Spontaneous Spatial Coupling of Body Movement for Touchless Pointing"** based on the following reasons:

> [!NOTE]
> **What is MatchPoint?**
> The technique works by displaying interactive widgets on the screen that perform continuous orbital motions. To select or interact with a widget, the user mimics the widget's motion with any part of their body (e.g., hand, head, or even a handheld object). By correlating the optical flow or coordinate trajectory of the tracked body part with the widget's movement, the system couples them, initiating pointing/selection without training classifiers or detecting specific body parts.

1. **Practical Hardware Requirements**:
   Unlike many papers that require specialized hardware (like eye-trackers, thermal cameras, biosensors, or custom wearables), *MatchPoint* only requires a **standard RGB Web Camera**. This makes it easy to implement and test on our local development laptops.

2. **Feasible & Well-Suited Scope**:
   * **No Machine Learning Training Required**: Because the system relies on *spatial coupling* (calculating the correlation between the movement trajectories of widgets and the user's tracking points), we do not need to collect large datasets or train complex deep learning classifiers.
   * **Focus on Core Interaction**: Within two weeks, we can build a robust computer-vision tracker (using open-source libraries like OpenCV/MediaPipe in Python or tracking libraries in JavaScript) that extracts motion vectors/flow and correlates them with our UI widgets' orbital paths.
   * This is a complete, self-contained project that is neither too simple (which would fail to demonstrate complex interaction implementation) nor too long/complex (which would be risky for a two-week timeline).

3. **High Novelty & User Experience**:
   It is a touchless, spontaneous interaction technique. Users can pick it up instantly, and it accommodates arbitrary body movements (head nod, hand wave, leg kick) to interact. It is highly demonstrative for a live presentation.

---

#### Bibliography / Citations

1. **Selected Paper**:
   * Clarke, C., & Gellersen, H. (2017). **MatchPoint: Spontaneous Spatial Coupling of Body Movement for Touchless Pointing**. In *Proceedings of the 2017 ACM International Conference on Interactive Surfaces and Spaces* (and presented at *UIST '17*). [Link to Lancaster University Eprints](https://eprints.lancs.ac.uk/id/eprint/88665/).

2. **Alternative Papers Considered**:
   * Liu, Y., Shan, Q., Yao, Z., & Lu, Q. (2024). **KeyFlow: Acoustic Motion Sensing for Cursor Control on Any Keyboard**. In *Adjunct Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology (UIST '24)*. [https://doi.org/10.1145/3654777.3676452](https://doi.org/10.1145/3654777.3676452).
   * Dachselt, R., & Buchholz, R. (2009). **Natural Throw and Tilt Interaction between Mobile Phones and Distant Displays**. In *CHI '09 Extended Abstracts on Human Factors in Computing Systems*. [https://doi.org/10.1145/1520340.1520448](https://doi.org/10.1145/1520340.1520448).
   * Latreche, N., & Schiettecatte, B. (2026). **Mapping Gestures Performed with Tangible Cubes to System Commands**. *Draft/Preprint on Tangible User Interfaces*.
   * Onishi, R., Morisaki, T., Suzuki, S., Mizutani, S., Kamigaki, T., Fujiwara, M., Makino, Y., & Shinoda, H. (2022). **GazeBreath: Input Method Using Gaze Pointing and Breath Selection**. In *Augmented Humans International Conference (AHs '22)*. [https://doi.org/10.1145/3519391.3519398](https://doi.org/10.1145/3519391.3519398).
   * Koo, D., O’Neill, T. C., Dinçer, S. B., Kwok, H. K. B., & Renelus, F. (2022). **Immersive Emotions: Translating Emotions Into Visualization**. In *Late-Breaking Results of the 24th International Conference on Human-Computer Interaction with Mobile Devices and Services (MobileHCI '22)*. [https://doi.org/10.1145/3543174.3546083](https://doi.org/10.1145/3543174.3546083).
   * Hall, B. D., Bartram, L., & Brehmer, M. (2022). **Augmented Chironomia for Presenting Data to Remote Audiences**. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology (UIST '22)*. [https://doi.org/10.1145/3526113.3545614](https://dl.acm.org/doi/10.1145/3526113.3545614).
   * Bilius, L. A., & Vatavu, R. D. (2021). **Demonstration of GestuRING, a Web Tool for Ring Gesture Input**. In *Adjunct Publication of the 34th Annual ACM Symposium on User Interface Software and Technology (UIST '21 Adjunct)*. [https://doi.org/10.1145/3474349.3480211](https://doi.org/10.1145/3474349.3480211).

---

### Task 2: Implementation

The replication is a single Python file (`implementation/matchpoint.py`) built on **OpenCV**, **NumPy**, and **SciPy**.  
Run it with:

```bash
pip3 install -r implementation/requirements.txt
python3 implementation/matchpoint.py
```

Controls: `d` debug overlay · `r` reset · `q` quit

---

#### Architecture

The application follows a two-state machine:

```
state = 'matching'  →  shows orbit, runs motion matcher
state = 'coupled'   →  runs ROI tracker, shows cursor + 4 demo targets
```

---

#### Phase 1 – Motion Matching

**Orbit widget** (`class Orbit`)  
A ring (radius 55 px, period 2.5 s) with a dot orbiting at constant angular velocity. The dot's x-position is a cosine wave and y-position a sine wave – these are the reference signals for correlation.

**Feature tracking** (`class FeatureTracker`)  
- FAST feature detector (threshold 20, non-max suppression ON) detects keypoints each frame.
- Pyramidal Lucas-Kanade optical flow (21×21 window, 3 levels) propagates each keypoint from frame to frame.
- Each feature maintains a `deque` of `HIST_LEN = 38` positions (≈ 1.25 s at 30 fps), matching the paper's requirement to capture at least half a revolution of the orbit.
- Points are re-detected periodically (every 15 frames) and when the tracked count falls below 60.

**Correlation check** (`class MotionMatcher`)  
For every feature with a full trajectory history:
1. Compute the Orbit's expected x- and y-trajectories for the same window using the known period (cosine/sine).
2. Z-normalise both the feature trajectory and the reference signal (Pearson r is scale-invariant).
3. Compute `pearsonr(xs_norm, exp_x_norm)` and `pearsonr(ys_norm, exp_y_norm)`.
4. If `max(|rx|, |ry|) ≥ 0.75` → candidate (paper threshold).

**RANSAC circle fit** (`fit_circle_ransac`)  
- 120 RANSAC iterations; 3-point minimal sample → analytic circle; inlier distance 8 px.
- Accepts fit if ≥ 6 inliers (paper: `RANSAC_MIN_IN`).
- `arc_coverage` bins trajectory angles into 16 sectors; requires ≥ 22 % coverage (at least ≈ 80° arc) – confirms the user traced a circle, not a straight line.

---

#### Phase 2 – ROI Initialisation

`initialise_roi(prev_gray, curr_gray, histories, …)`

1. Compute the per-feature displacement vectors `a_i = prev – curr` (Eq. 1 in the paper).
2. Calculate the average unit vector → average angle θ̄; find magnitude range D_min, D_max.
3. Compute **dense Farnebäck optical flow** between the last two frames.
4. Build a pixel mask: keep pixels where  
   `(1−0.25)·D_min ≤ |flow| ≤ (1+0.25)·D_max`  (Eq. 6)  
   and  `|∠flow − θ̄| < π/8`  (Eq. 7).
5. **Connected-component labelling** seeded with 10×10 px areas around each matched feature point (paper: avoids offset errors from individual feature tracking).
6. Fit axis-aligned bounding box around the component → initial ROI.  
   Fallback (no component): 30×30 px box around the seed point.

**CD-gain** (Eq. 8):  
`CDgain = (1 / r_fit) × screen_dim`  
where `r_fit` is the radius of the RANSAC-fitted circle in camera pixels. This makes the gain inversely proportional to how large the user's matching motion was, normalising for distance from camera and body part used.

---

#### Phase 3 – Modified Median Flow Tracker

`class MedianFlowTracker`

**Grid generation** (`_caterer_grid`)  
Tracking points are placed on a grid with spacing `sqrt(w×h / 25)`, inspired by the lazy caterer's sequence mentioned in the paper. This concentrates points toward the centre of the ROI, reducing the chance of latching onto background pixels.

**Median Flow step**  
Each frame: track the grid with LK optical flow → take the **median** of all displacement vectors → translate ROI by that amount.

**Periodic recalibration**  
Every `RECALIB_INT = 0.5 s` (and when ROI moved > 2 px): re-run dense Farnebäck OF in a 2× search region; re-fit bounding box using connected components seeded from the current ROI centre. Record the offset between the old and new ROI centre so the cursor position does not jump (paper: *"the recalibration is unnoticeable to the user"*).

**Jitter filter** (Eq. 9)  
When frame-to-frame cursor movement `dt < dMIN = 2.5 px`:  
`NB = NMAX − ⌊dt × NMAX / dMIN⌋`  (dynamic window size, up to 10 frames)  
The cursor position is the mean of the last `NB` positions. This smooths out image noise without adding perceptible lag during fast movement.

**Cursor mapping**  
`cursor_x = start_x + (roi_cx − initial_roi_cx) × CDgain`  
`cursor_y = start_y + (roi_cy − initial_roi_cy) × CDgain`  
The cursor starts at the display position corresponding to the initial ROI centre; subsequent movement is amplified by the CD-gain.

**Pointer termination**  
If ROI centre has not moved > 0.3 px for `IDLE_TIMEOUT = 3 s` → decouple, return to matching state.

---

#### Demo Targets

Four circular targets (A, B, C, D) are arranged in a cross pattern (radius 290 px from centre). Hovering the cursor over a target for `DWELL_TIME = 0.8 s` activates it (fills green) with a progress arc indicator. This demonstrates the pointing accuracy achievable with the technique.

---

#### Design Decisions & Challenges

| Decision | Rationale |
|---|---|
| Single orbit, single pointer | The paper's core contribution is the coupling mechanism; one orbit cleanly demonstrates it without UI overhead. |
| Mirrored camera frame | Mirroring (flip horizontal) before processing makes the interaction feel natural – left/right movement maps as expected. |
| `HIST_LEN = 38` at 30 fps | Covers ≈ 1.25 s = 0.5 orbital periods; sufficient to detect the circular pattern but short enough to react quickly. |
| Pearson over raw cross-correlation | Scale-invariant: the user's hand 5 cm from camera and 50 cm from camera produce equal correlation scores. |
| Cursor starts at ROI position, not screen centre | Avoids a large initial cursor jump; the cursor appears where the matched body part is visible. |
| Dense OF recalibration threshold 0.5 s / 2 px | Paper values verbatim. Balances tracking accuracy against CPU cost (dense OF is ~3× more expensive per frame). |

**Challenges encountered:**
- Feature points on low-texture backgrounds (e.g., plain walls) yield noisy trajectories → mitigated by requiring `MIN_DISP_PX = 2` and the arc coverage check.
- RANSAC can fail on very short HIST_LEN windows → resolved by setting `HIST_LEN = 38` (> half a period).
- Pearson correlation phase shift: if the user lags the orbit by > 60°, `|r| < 0.5` → they need to sync with the dot within ≈ ±60° phase. This is natural once the user understands the mechanic.

---

#### Setup

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| opencv-python | ≥ 4.5 (tested: 5.0.0) |
| numpy | ≥ 1.20 (tested: 2.5.0) |
| scipy | ≥ 1.7 (tested: 1.18.0) |

```bash
# Install
pip3 install -r implementation/requirements.txt

# Run
python3 implementation/matchpoint.py
```

Webcam device 0 is used. The window opens at 1280 × 720 px.  
Point any webcam-visible part of your body at the screen, then move it in a circle matching the orbiting dot. After ≈ 1 second of consistent matching, the cursor appears.