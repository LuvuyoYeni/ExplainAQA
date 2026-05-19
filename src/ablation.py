import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

from load_dataset import load_uiprmd, select_features, N_SAMPLES_PER_CLASS
from tcn_model import TCNQualityRegressor, LSTMQualityRegressor
from train import subject_folds, run_epoch, evaluate, DEVICE, BATCH_SIZE

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

ABLATION_EPOCHS = 100


def dtw_align_dataset(X: np.ndarray, reference_idx: int = 0) -> np.ndarray:
    from dtw import dtw as _dtw, warp_to_reference

    reference = X[reference_idx].astype(np.float64)
    X_aligned = np.zeros_like(X)

    for i in range(len(X)):
        path, _ = _dtw(reference, X[i].astype(np.float64), band_ratio=0.1)
        X_aligned[i] = warp_to_reference(X[i].astype(np.float64), path)

    return X_aligned.astype(np.float32)


def _run_one_fold(model, X_tr_t, y_tr_t, X_te_t, y_te_t, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=epochs)
    criterion = nn.MSELoss()

    best_val, best_state = float("inf"), None

    for _ in range(epochs):
        run_epoch(model, X_tr_t, y_tr_t, optimizer, criterion)
        val_mse, _, _ = evaluate(model, X_te_t, y_te_t)
        scheduler.step()
        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, mae, rho = evaluate(model, X_te_t, y_te_t)
    return mae, rho


def run_config(X_data, scores, folds, model_cls, model_kwargs,
               epochs=ABLATION_EPOCHS):
    fold_rhos, fold_maes = [], []

    for train_idx, test_idx in folds:
        X_tr = torch.tensor(X_data[train_idx], dtype=torch.float32)
        X_te = torch.tensor(X_data[test_idx],  dtype=torch.float32)
        y_tr = torch.tensor(scores[train_idx],  dtype=torch.float32)
        y_te = torch.tensor(scores[test_idx],   dtype=torch.float32)

        model = model_cls(**model_kwargs).to(DEVICE)
        mae, rho = _run_one_fold(model, X_tr, y_tr, X_te, y_te, epochs)
        fold_maes.append(mae)
        fold_rhos.append(rho)

    return (float(np.mean(fold_rhos)), float(np.std(fold_rhos)),
            float(np.mean(fold_maes)), float(np.std(fold_maes)))


def main():
    print("Loading data …")
    X_raw, _, scores = load_uiprmd()
    X_angles = select_features(X_raw)

    print(f"\nAligning {len(X_angles)} sequences with DTW "
          f"(reference = sample 0, correct) …")
    t0 = time.time()
    X_aligned = dtw_align_dataset(X_angles, reference_idx=0)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    folds = subject_folds()

    configs = [
        (
            "Raw coords  +  LSTM",
            X_raw,
            LSTMQualityRegressor,
            {"in_features": 117, "hidden_size": 64},
        ),
        (
            "Raw coords  +  TCN",
            X_raw,
            TCNQualityRegressor,
            {"in_features": 117, "num_channels": 64},
        ),
        (
            "Angles (no DTW) + LSTM",
            X_angles,
            LSTMQualityRegressor,
            {"in_features": 7, "hidden_size": 64},
        ),
        (
            "Angles (no DTW) + TCN",
            X_angles,
            TCNQualityRegressor,
            {"in_features": 7, "num_channels": 64},
        ),
        (
            "DTW-aligned + LSTM",
            X_aligned,
            LSTMQualityRegressor,
            {"in_features": 7, "hidden_size": 64},
        ),
        (
            "DTW-aligned + TCN",
            X_aligned,
            TCNQualityRegressor,
            {"in_features": 7, "num_channels": 64},
        ),
    ]

    print(f"{'Configuration':<26}  {'Spearman ρ':>18}  {'MAE':>14}  {'Time':>6}")
    print("-" * 72)

    results = []
    for label, X_data, model_cls, mkw in configs:
        t0 = time.time()
        rho_m, rho_s, mae_m, mae_s = run_config(
            X_data, scores, folds, model_cls, mkw)
        elapsed = time.time() - t0
        results.append((label, rho_m, rho_s, mae_m, mae_s))
        print(f"{label:<26}  {rho_m:+.4f} ± {rho_s:.4f}  "
              f"{mae_m:.4f} ± {mae_s:.4f}  {elapsed:5.1f}s")

    print()
    print("=" * 72)
    print("ABLATION RESULTS  (5-fold subject-independent CV, 100 epochs each)")
    print("=" * 72)
    print(f"{'Configuration':<26}  {'Spearman ρ (μ ± σ)':>20}  {'MAE (μ ± σ)':>16}")
    print("-" * 72)
    for label, rho_m, rho_s, mae_m, mae_s in results:
        bar = "◀ best ρ" if rho_m == max(r[1] for r in results) else ""
        print(f"{label:<26}  {rho_m:+.4f} ± {rho_s:.4f}       "
              f"{mae_m:.4f} ± {mae_s:.4f}  {bar}")
    print("=" * 72)


if __name__ == "__main__":
    main()
