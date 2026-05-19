import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from load_dataset import load_uiprmd, select_features, JOINT_NAMES
from tcn_model import TCNQualityRegressor
from explain import compute_attribution, check_constraints, generate_feedback

OUTPUTS = Path(__file__).parent.parent / "outputs"
DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    model = TCNQualityRegressor(in_features=7).to(DEVICE)
    ckpt  = torch.load(OUTPUTS / "best_model.pt", map_location=DEVICE,
                       weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded from {OUTPUTS / 'best_model.pt'}  (device: {DEVICE})")

    print("Loading data …")
    X_raw, y, scores = load_uiprmd()
    X = select_features(X_raw)
    print()

    correct_idx   = int(np.where(y == 1)[0][0])
    incorrect_idx = int(np.where(y == 0)[0][0])

    samples = [
        ("CORRECT",   correct_idx,   X[correct_idx],   scores[correct_idx]),
        ("INCORRECT", incorrect_idx, X[incorrect_idx], scores[incorrect_idx]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    for ax, (label, idx, seq, true_score) in zip(axes, samples):
        x_in = seq[np.newaxis]

        fb   = generate_feedback(model, x_in, JOINT_NAMES)
        attr = compute_attribution(model, x_in)

        print(f"{'='*62}")
        print(f"Sample {idx:3d}  [{label}]   true score = {true_score:.4f}")
        print(f"  Predicted score : {fb['predicted_score']:.4f}")
        print(f"  Top-3 most negative attributions:")
        for joint, frame, val in fb["top_attributions"]:
            print(f"    {joint:<22s}  frame {frame:3d}   gradient {val:+.6f}")
        print(f"  Constraint violations: {len(fb['constraint_violations'])}")
        for v in fb["constraint_violations"]:
            print(f"    [{v['rule']}]  {v['description']}")
        print()
        print(f"  FEEDBACK:")
        print(f"  {fb['feedback_text']}")
        print()

        vmax = max(np.abs(attr).max(), 1e-9)
        im = ax.imshow(
            attr.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_yticks(range(7))
        ax.set_yticklabels(JOINT_NAMES, fontsize=9)
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_title(
            f"{label} (sample {idx})  |  true = {true_score:.4f}   "
            f"pred = {fb['predicted_score']:.4f}  |  "
            f"{len(fb['constraint_violations'])} violation(s)",
            fontsize=9,
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="∂score/∂input")

        lk         = seq[:, 0]
        rk         = seq[:, 1]
        avg_k      = (lk + rk) / 2.0
        peak_frame = int(np.argmin(avg_k))

        marked_rules = set()
        for v in fb["constraint_violations"]:
            rule = v["rule"]
            if rule in ("peak_knee_flexion", "bilateral_knee_asymmetry",
                        "descent_duration", "knee_hyperextension") \
                    and "peak" not in marked_rules:
                ax.axvline(x=peak_frame, color="red", linewidth=1.5,
                           linestyle="--", alpha=0.85,
                           label=f"peak depth f{peak_frame}")
                marked_rules.add("peak")
            elif rule == "pelvis_obliquity":
                po_frames = np.where(seq[:, 6] >= 20.0)[0]
                for f in po_frames[::4]:
                    ax.axvline(x=f, color="red", linewidth=0.6,
                               alpha=0.35)
                if len(po_frames):
                    ax.axvline(x=po_frames[0], color="red", linewidth=0.6,
                               alpha=0.35, label="pelvis > 20°")
                    marked_rules.add("pelvis")

        if marked_rules:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("ExplainAQA — Gradient Attribution Maps (∂score/∂joint angle)",
                 fontsize=11)
    plt.tight_layout()
    out_path = OUTPUTS / "attribution_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Attribution heatmap saved → {out_path}")


if __name__ == "__main__":
    main()
