# === OCR / pipeline configuration ===
OCR_FRAME_GAP = 1
MIN_PLAYER_H = 50
AREA_GROWTH_TRIG = 1.4
CONF_MIN = 0.60
LEAKY_DECAY = 0.001

DRAW_ROI = True
ROI_COLORS = {"torso": (0, 255, 255)}  # BGR
ROI_THICK = 2

LEGIBILITY_ENABLED = True
LEG_THR = 0.75
LEG_MIN_H, LEG_MIN_W = 40, 40

USE_PARSEQ = True
PARSEQ_CKPT = "ocrModels/parseq_tiny_epoch=4-step=665-val_accuracy=96.5800-val_NED=97.8600.ckpt"

# Colors / classes for drawing
COLORS = {2: (0, 0, 255), 3: (255, 255, 0)}  # Ball, Ref
TARGET_CLASSES = [0, 1, 2, 3]

# Voting / hysteresis
VOTE_MIN_TOTAL_CONF = 1.5
ONE_DIGIT_PENALTY = 0.8
HYSTERESIS_MARGIN = 1.2
DELTA_MIN = 0.30
USE_LEAKY = True
