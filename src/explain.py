import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

FEATURE_NAMES = [
    "left_knee", "right_knee", "left_hip", "right_hip",
    "left_ankle", "right_ankle", "pelvis_obliquity",
]

LK, RK, LH, RH, LA, RA, PO = range(7)


def compute_attribution(model, x):
    device = next(model.parameters()).device
    model.eval()

    if isinstance(x, np.ndarray):
        x_t = torch.tensor(x, dtype=torch.float32)
    else:
        x_t = x.float().clone()

    x_t = x_t.to(device)
    x_t.requires_grad_(True)

    score = model(x_t)
    score.sum().backward()

    return x_t.grad.detach().cpu().numpy()[0]


def check_constraints(sequence):
    lk = sequence[:, LK]
    rk = sequence[:, RK]
    lh = sequence[:, LH]
    rh = sequence[:, RH]
    po = sequence[:, PO]

    avg_k      = (lk + rk) / 2.0
    peak_frame = int(np.argmin(avg_k))
    T          = len(lk)

    violations = []

    min_lk = float(np.min(lk))
    min_rk = float(np.min(rk))
    worst  = max(min_lk, min_rk)
    side   = "left" if min_lk > min_rk else "right"
    v1     = worst > 90.0
    violations.append({
        "rule": "peak_knee_flexion",
        "violated": v1,
        "measured_value": round(worst, 1),
        "threshold": 90.0,
        "description": (
            f"Peak {side} knee flexion was insufficient "
            f"(minimum angle {worst:.1f} deg; must reach <= 90 deg)."
        ) if v1 else "",
    })

    asym = float(abs(lk[peak_frame] - rk[peak_frame]))
    v2   = asym > 15.0
    violations.append({
        "rule": "bilateral_knee_asymmetry",
        "violated": v2,
        "measured_value": round(asym, 1),
        "threshold": 15.0,
        "description": (
            f"Bilateral knee asymmetry of {asym:.1f} deg at peak depth "
            f"(frame {peak_frame}); limit is 15 deg."
        ) if v2 else "",
    })

    max_po = float(np.max(po))
    v3     = max_po >= 20.0
    violations.append({
        "rule": "pelvis_obliquity",
        "violated": v3,
        "measured_value": round(max_po, 1),
        "threshold": 20.0,
        "description": (
            f"Excessive pelvis tilt detected (peak {max_po:.1f} deg; limit 20 deg)."
        ) if v3 else "",
    })

    hip_range = max(float(np.max(lh) - np.min(lh)), float(np.max(rh) - np.min(rh)))
    v4        = hip_range < 20.0
    violations.append({
        "rule": "hip_flexion",
        "violated": v4,
        "measured_value": round(hip_range, 1),
        "threshold": 20.0,
        "description": (
            f"Insufficient hip flexion (range {hip_range:.1f} deg; need >= 20 deg)."
        ) if v4 else "",
    })

    prev_stand = 0
    for i in range(1, peak_frame - 1):
        if avg_k[i] >= avg_k[i-1] and avg_k[i] >= avg_k[i+1]:
            prev_stand = i
            break
    next_stand = peak_frame + 1
    for i in range(peak_frame + 1, T - 1):
        if avg_k[i] >= avg_k[i-1] and avg_k[i] >= avg_k[i+1]:
            next_stand = i
            break
    rep_len     = max(next_stand - prev_stand, 1)
    desc_frames = max(peak_frame - prev_stand, 0)
    desc_ratio  = desc_frames / rep_len
    v5          = desc_ratio < 0.4
    violations.append({
        "rule": "descent_duration",
        "violated": v5,
        "measured_value": round(desc_ratio, 3),
        "threshold": 0.4,
        "description": (
            f"Descent too short ({desc_frames} frames, {desc_ratio:.1%} of rep); "
            f"should be >= 40%."
        ) if v5 else "",
    })

    overall_min = min(min_lk, min_rk)
    h_side      = "left" if min_lk < min_rk else "right"
    v6          = overall_min < 5.0
    violations.append({
        "rule": "knee_hyperextension",
        "violated": v6,
        "measured_value": round(overall_min, 1),
        "threshold": 5.0,
        "description": (
            f"Extreme {h_side} knee angle ({overall_min:.1f} deg); "
            f"must stay above 5 deg."
        ) if v6 else "",
    })

    return violations


def generate_feedback(model, x, feature_names=None):
    if feature_names is None:
        feature_names = FEATURE_NAMES

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x_t  = torch.tensor(x, dtype=torch.float32).to(device)
        pred = max(0.0, min(1.0, float(model(x_t).item())))

    attr = compute_attribution(model, x)
    seq  = x[0]

    flat     = attr.flatten()
    neg_idx  = np.argsort(flat)[:3]
    top_attr = [(feature_names[int(i % 7)], int(i // 7), float(flat[i])) for i in neg_idx]

    all_viols = check_constraints(seq)
    violated  = [v for v in all_viols if v["violated"]]

    parts = [f"Predicted quality score: {pred:.3f}."]
    if violated:
        for v in violated:
            parts.append(v["description"])
    else:
        parts.append("No hard constraint violations detected.")

    top_joint, top_frame, top_val = top_attr[0]
    if top_val < -0.005:
        parts.append(
            f"Largest model concern: {top_joint} at frame {top_frame} "
            f"(gradient {top_val:+.5f})."
        )

    return {
        "predicted_score":       pred,
        "top_attributions":      top_attr,
        "constraint_violations": violated,
        "feedback_text":         " ".join(parts),
    }
