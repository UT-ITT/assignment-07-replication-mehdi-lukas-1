import numpy as np

# ----------
# TraceMatch
# ----------
DEDUPE_WIN = 10
MOTION_THRESHOLD = 4.0
PEARSON_THRESH = 0.98   # |Pearson r| threshold for accepting a match
MIN_ARC        = 0.22   # minimum fraction of circle arc that must be covered
RANSAC_N_ITER  = 120    # RANSAC iterations for circle fit
RANSAC_MIN_IN  = 6      # minimum inliers to accept a circle fit
MIN_DISP_PX    = 2.0    # minimum total displacement to consider a feature (px)
THIN           = 0.03

# -------
# Tracking
# -------
TH_D    = 0.25           # distance tolerance for dense-OF pixel match (Eq. 6)
TH_ANGL = np.pi / 8      # angle tolerance ≈ 22.5° (Eq. 7)
MIN_ROI = 30             # fallback minimum ROI side length (px)

RECALIB_MOVE = 2.0       # minimum ROI displacement to trigger recalibration (px)
RECALIB_INT  = 0.5       # minimum seconds between recalibrations
JITTER_DMIN  = 2.5       # motion below this activates jitter filter (px)  [paper: 2.5]
JITTER_NMAX  = 10        # maximum moving-window size for jitter filter      [paper: 10]

WIN_W, WIN_H = 1280, 720 # display window size (px)
IDLE_TIMEOUT = 3.0       # seconds of stillness → decouple
DWELL_TIME = 0.8         # seconds to dwell over a target to activate it

# -------
# Widgets
# -------
# seconds of stillness -> decouple
IDLE_TIMEOUT = 3.0
# seconds to dwell over a target to activate it
DWELL_TIME = 0.8

# ------
# COLORS
# ------

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
