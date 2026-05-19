import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from pathlib import Path

from app import annotate_video

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

W, H   = 640, 480
FPS    = 30.0
N      = 60
JOINTS = 7

frames_bgr = []
for fi in range(N):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :, 0] = int(fi / N * 180)
    frame[:, :, 1] = 40
    frames_bgr.append(frame)

lm_per_frame = []
for fi in range(N):
    lms = []
    for k in range(33):
        t = fi / N
        lms.append([0.3 + 0.4 * np.sin(t * np.pi + k * 0.3),
                    0.1 + 0.7 * (k / 32.0),
                    0.0])
    lm_per_frame.append(lms)

t240 = np.linspace(0, 2 * np.pi, 240)
resampled = np.stack([
    90 + 60 * np.sin(t240),
    90 + 55 * np.sin(t240 + 0.1),
    120 + 30 * np.sin(t240 + 0.5),
    120 + 28 * np.sin(t240 + 0.5),
    100 + 20 * np.cos(t240),
    100 + 20 * np.cos(t240 + 0.1),
    15 + 10 * np.sin(t240 * 2),
], axis=1).astype(np.float32)

rng  = np.random.default_rng(0)
attr = rng.normal(0, 0.0002, (240, 7)).astype(np.float32)
attr[80, 0] = -0.012
attr[80, 1] = -0.008

feedback = {
    "predicted_score": 0.72,
    "constraint_violations": [
        {
            "rule": "bilateral_knee_asymmetry",
            "violated": True,
            "measured_value": 22.5,
            "threshold": 15.0,
            "description": "Bilateral knee asymmetry of 22.5 deg at peak depth.",
        },
        {
            "rule": "pelvis_obliquity",
            "violated": True,
            "measured_value": 23.1,
            "threshold": 20.0,
            "description": "Excessive pelvis tilt detected (peak 23.1 deg).",
        },
    ],
    "top_attributions": [("left_knee", 80, -0.012)],
    "feedback_text": "Synthetic test feedback.",
}

out_path = OUTPUTS / "annotated_output.mp4"

print(f"Writing synthetic annotated video → {out_path}")
print(f"  {N} frames  {W}x{H}  {FPS} fps")
annotate_video(frames_bgr, lm_per_frame, resampled,
               attr, feedback, FPS, (W, H), out_path)

cap       = cv2.VideoCapture(str(out_path))
ok        = cap.isOpened()
n_written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w_out     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_out     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out   = cap.get(cv2.CAP_PROP_FPS)
cap.release()

file_kb = out_path.stat().st_size / 1024

print()
print("── Verification ─────────────────────────────────────")
print(f"  File exists:    {out_path.exists()}")
print(f"  File size:      {file_kb:.1f} KB")
print(f"  Readable:       {ok}")
print(f"  Frames written: {n_written}  (expected {N})")
print(f"  Resolution:     {w_out}x{h_out}  (expected {W}x{H})")
print(f"  FPS:            {fps_out:.1f}  (expected {FPS})")

assert ok,         "VideoCapture could not open the output file"
assert n_written >= N - 1, f"Expected ~{N} frames, got {n_written}"
assert w_out == W, f"Width mismatch: {w_out} != {W}"
assert h_out == H, f"Height mismatch: {h_out} != {H}"

print()
print("✓  annotate_video() produced a valid MP4.")
