import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from scipy.stats import spearmanr

from load_dataset import load_uiprmd, select_features, N_SAMPLES_PER_CLASS
from dtw import dtw
from tcn_model import TCNQualityRegressor
from train import subject_folds, evaluate, DEVICE, BATCH_SIZE

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def enroll_gallery(X_train, y_train):
    correct_idx = np.where(y_train == 1)[0]
    templates   = X_train[correct_idx]

    ref = templates[0]
    acc = ref.copy().astype(np.float64)
    for i in range(1, len(templates)):
        q    = templates[i].astype(np.float64)
        path, _ = dtw(ref, q)
        ref_len = path[-1][0] + 1
        buckets = [[] for _ in range(ref_len)]
        for ri, qi in path:
            buckets[ri].append(qi)
        warped = np.array([np.mean(q[idxs], axis=0) for idxs in buckets])
        acc += warped
    mean_template = (acc / len(templates)).astype(np.float32)

    return {
        "template":      mean_template,
        "all_templates": templates,
    }


def verify(probe, gallery, threshold):
    template = gallery["template"].astype(np.float64)
    p        = probe.astype(np.float64)
    _, cost  = dtw(template, p)
    accept   = cost <= threshold
    return accept, float(cost)


def compute_biometric_metrics(X, y, gallery, thresholds):
    correct_idx   = np.where(y == 1)[0]
    incorrect_idx = np.where(y == 0)[0]

    genuine_scores   = []
    impostor_scores  = []

    print("  Computing genuine scores...")
    for i in correct_idx:
        _, score = verify(X[i], gallery, threshold=np.inf)
        genuine_scores.append(score)

    print("  Computing impostor scores...")
    for i in incorrect_idx:
        _, score = verify(X[i], gallery, threshold=np.inf)
        impostor_scores.append(score)

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    far_arr = []
    frr_arr = []

    for thr in thresholds:
        fa  = np.sum(impostor_scores <= thr)
        fr  = np.sum(genuine_scores  >  thr)
        far_arr.append(fa / len(impostor_scores))
        frr_arr.append(fr / len(genuine_scores))

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)

    diff    = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)

    return {
        "far":            far_arr,
        "frr":            frr_arr,
        "thresholds":     thresholds,
        "eer":            eer,
        "eer_threshold":  float(thresholds[eer_idx]),
        "far_at_eer":     float(far_arr[eer_idx]),
        "frr_at_eer":     float(frr_arr[eer_idx]),
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
    }


def run_pipeline1_biometrics(X, y, scores, thresholds):
    model = TCNQualityRegressor(in_features=7).to(DEVICE)
    model.load_state_dict(torch.load(OUTPUTS / "best_model.pt", map_location=DEVICE))
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    for start in range(0, len(X_t), BATCH_SIZE):
        xb = X_t[start:start + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            preds.append(model(xb).cpu().numpy())
    pred_scores = np.concatenate(preds)

    correct_idx   = np.where(y == 1)[0]
    incorrect_idx = np.where(y == 0)[0]

    genuine_scores   = pred_scores[correct_idx]
    impostor_scores  = pred_scores[incorrect_idx]

    rho, _ = spearmanr(pred_scores, scores)
    mae    = float(np.mean(np.abs(pred_scores - scores)))

    far_arr = []
    frr_arr = []

    for thr in thresholds:
        fa  = np.sum(impostor_scores >= thr)
        fr  = np.sum(genuine_scores  <  thr)
        far_arr.append(fa / len(impostor_scores))
        frr_arr.append(fr / len(genuine_scores))

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)

    diff    = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)

    return {
        "rho":       float(rho),
        "mae":       mae,
        "eer":       eer,
        "far_at_eer": float(far_arr[eer_idx]),
        "frr_at_eer": float(frr_arr[eer_idx]),
        "far":       far_arr,
        "frr":       frr_arr,
        "thresholds": thresholds,
    }


def cross_val_pipeline2(X, y):
    folds   = subject_folds()
    results = []

    print("Estimating DTW score range for threshold grid...")
    gallery_full  = enroll_gallery(X[np.where(y == 1)[0]], np.ones(int(y.sum())))
    sample_scores = []
    for i in range(min(20, len(X))):
        _, s = verify(X[i], gallery_full, threshold=np.inf)
        sample_scores.append(s)
    lo = min(sample_scores) * 0.5
    hi = max(sample_scores) * 1.5
    thresholds = np.linspace(lo, hi, 200)
    print(f"  Threshold range: [{lo:.1f}, {hi:.1f}]")
    print()

    all_eer = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        print(f"Fold {fi+1}/5  train={len(tr_idx)}  test={len(te_idx)}")

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test,  y_test  = X[te_idx], y[te_idx]

        print("  Enrolling gallery...")
        gallery = enroll_gallery(X_train, y_train)

        metrics = compute_biometric_metrics(X_test, y_test, gallery, thresholds)
        eer = metrics["eer"]
        all_eer.append(eer)
        print(f"  EER={eer:.4f}  FAR@EER={metrics['far_at_eer']:.4f}  FRR@EER={metrics['frr_at_eer']:.4f}")
        results.append(metrics)

    print()
    print(f"Mean EER: {np.mean(all_eer):.4f} +/- {np.std(all_eer):.4f}")

    return results, thresholds


def save_det_curve(p2_results, p1_results, thresholds):
    fig, ax = plt.subplots(figsize=(7, 6))

    far_mean = np.mean([r["far"] for r in p2_results], axis=0)
    frr_mean = np.mean([r["frr"] for r in p2_results], axis=0)
    ax.plot(far_mean, frr_mean, color="#1565C0", lw=2, label="Pipeline 2 (DTW auth)")

    ax.plot(p1_results["far"], p1_results["frr"], color="#C62828", lw=2,
            linestyle="--", label="Pipeline 1 (TCN quality)")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)

    p2_eer = np.mean([r["eer"] for r in p2_results])
    ax.scatter([p2_eer], [p2_eer], color="#1565C0", s=80, zorder=5)
    ax.scatter([p1_results["eer"]], [p1_results["eer"]], color="#C62828", s=80, zorder=5)

    ax.set_xlabel("FAR (False Accept Rate)", fontsize=11)
    ax.set_ylabel("FRR (False Reject Rate)", fontsize=11)
    ax.set_title("DET Curve: Pipeline 1 vs Pipeline 2", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "det_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"DET curve saved -> {OUTPUTS}/det_curve.png")


def save_comparison_csv(p2_results, p1_results, p2_rho=None, p2_mae=None):
    p2_eer     = float(np.mean([r["eer"]       for r in p2_results]))
    p2_far_eer = float(np.mean([r["far_at_eer"] for r in p2_results]))
    p2_frr_eer = float(np.mean([r["frr_at_eer"] for r in p2_results]))

    rows = [
        ["Metric",            "Pipeline 1 (TCN)",           "Pipeline 2 (DTW auth)"],
        ["Spearman rho",      f"{p1_results['rho']:+.4f}",  "N/A"],
        ["MAE",               f"{p1_results['mae']:.4f}",   "N/A"],
        ["EER",               f"{p1_results['eer']:.4f}",   f"{p2_eer:.4f}"],
        ["FAR @ EER",         f"{p1_results['far_at_eer']:.4f}", f"{p2_far_eer:.4f}"],
        ["FRR @ EER",         f"{p1_results['frr_at_eer']:.4f}", f"{p2_frr_eer:.4f}"],
        ["Training required", "Yes",                          "No"],
        ["Provides explanation", "Yes (gradient attribution)", "No"],
    ]

    path = OUTPUTS / "pipeline_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Comparison table saved -> {path}")

    print()
    print("=" * 56)
    print(f"{'Metric':<24} {'Pipeline 1':>14} {'Pipeline 2':>14}")
    print("-" * 56)
    for row in rows[1:]:
        print(f"{row[0]:<24} {row[1]:>14} {row[2]:>14}")
    print("=" * 56)


def main():
    print("Loading data...")
    X_raw, y, scores = load_uiprmd()
    X = select_features(X_raw)
    print()

    print("=" * 50)
    print("PIPELINE 2: DTW-based movement authentication")
    print("=" * 50)
    p2_results, thresholds = cross_val_pipeline2(X, y)

    print()
    print("=" * 50)
    print("PIPELINE 1: TCN quality score as matching score")
    print("=" * 50)
    p1_results = run_pipeline1_biometrics(X, y, scores, np.linspace(0.0, 1.5, 200))
    print(f"  rho={p1_results['rho']:+.4f}  mae={p1_results['mae']:.4f}  "
          f"EER={p1_results['eer']:.4f}  FAR@EER={p1_results['far_at_eer']:.4f}  "
          f"FRR@EER={p1_results['frr_at_eer']:.4f}")

    print()
    save_det_curve(p2_results, p1_results, thresholds)
    save_comparison_csv(p2_results, p1_results)


if __name__ == "__main__":
    main()
