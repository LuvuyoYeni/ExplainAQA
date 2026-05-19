import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

from load_dataset import load_uiprmd, select_features, N_SAMPLES_PER_CLASS
from tcn_model import TCNQualityRegressor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
N_SUBJECTS   = 10
SPB          = N_SAMPLES_PER_CLASS // N_SUBJECTS


def subject_folds(n_folds=5):
    spf = N_SUBJECTS // n_folds
    folds = []
    for f in range(n_folds):
        test_subjs  = list(range(f * spf, (f + 1) * spf))
        train_subjs = [s for s in range(N_SUBJECTS) if s not in test_subjs]

        tr_cor = [i for s in train_subjs for i in range(s * SPB, (s + 1) * SPB)]
        te_cor = [i for s in test_subjs  for i in range(s * SPB, (s + 1) * SPB)]
        tr_idx = tr_cor + [i + N_SAMPLES_PER_CLASS for i in tr_cor]
        te_idx = te_cor + [i + N_SAMPLES_PER_CLASS for i in te_cor]
        folds.append((tr_idx, te_idx))
    return folds


def run_epoch(model, X_t, y_t, optimizer, criterion):
    model.train()
    n    = len(X_t)
    perm = torch.randperm(n)
    X_t, y_t = X_t[perm], y_t[perm]

    total = 0.0
    for start in range(0, n, BATCH_SIZE):
        xb = X_t[start:start + BATCH_SIZE].to(DEVICE)
        yb = y_t[start:start + BATCH_SIZE].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / n


@torch.no_grad()
def evaluate(model, X_t, y_t):
    model.eval()
    preds = []
    for start in range(0, len(X_t), BATCH_SIZE):
        xb = X_t[start:start + BATCH_SIZE].to(DEVICE)
        preds.append(model(xb).cpu())
    pred = torch.cat(preds).numpy()
    true = y_t.numpy()

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rho, _ = spearmanr(pred, true)
    return mse, mae, float(rho)


def train_fold(X_train, y_train, X_val, y_val, in_features=7):
    model = TCNQualityRegressor(in_features=in_features).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.float32)

    best_val, best_w = float("inf"), None
    tr_losses, vl_losses = [], []

    for ep in range(EPOCHS):
        tr_loss = run_epoch(model, X_tr, y_tr, opt, loss_fn)
        val_mse, _, _ = evaluate(model, X_vl, y_vl)
        sched.step()

        tr_losses.append(tr_loss)
        vl_losses.append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_w)
    return model, tr_losses, vl_losses


def main(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    X_raw, _, scores = load_uiprmd()
    X = select_features(X_raw)
    print()

    folds      = subject_folds()
    results    = []
    all_tr, all_vl = [], []
    best_rho   = -float("inf")
    best_model = None

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_train, y_train = X[tr_idx], scores[tr_idx]
        X_test,  y_test  = X[te_idx], scores[te_idx]
        print(f"Fold {fi+1}/5  train={len(tr_idx)}  test={len(te_idx)}")

        model, tr_l, vl_l = train_fold(X_train, y_train, X_test, y_test)
        _, mae, rho = evaluate(model,
                               torch.tensor(X_test, dtype=torch.float32),
                               torch.tensor(y_test, dtype=torch.float32))

        print(f"  rho={rho:+.4f}  mae={mae:.4f}")
        results.append({"fold": fi + 1, "rho": rho, "mae": mae})
        all_tr.append(tr_l)
        all_vl.append(vl_l)

        if rho > best_rho:
            best_rho   = rho
            best_model = model

    rhos = [r["rho"] for r in results]
    maes = [r["mae"] for r in results]
    print()
    print("=" * 44)
    print("5-Fold CV Summary")
    print(f"  rho: {np.mean(rhos):+.4f} +/- {np.std(rhos):.4f}")
    print(f"  mae:  {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
    print("=" * 44)

    torch.save(best_model.state_dict(), OUTPUTS / "best_model.pt")
    print(f"\nBest model (rho={best_rho:+.4f}) saved -> {OUTPUTS}/best_model.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for fi in range(5):
        axes[0].plot(all_tr[fi], color=colors[fi], alpha=0.8, label=f"Fold {fi+1}")
        axes[1].plot(all_vl[fi], color=colors[fi], alpha=0.8, label=f"Fold {fi+1}")
    for ax, title in zip(axes, ["Train MSE", "Val MSE"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("TCN Training Curves (5-fold CV)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved -> {OUTPUTS}/training_curves.png")

    return results


if __name__ == "__main__":
    main()
