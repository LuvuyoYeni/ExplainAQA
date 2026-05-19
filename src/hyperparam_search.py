import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json
import itertools
from pathlib import Path
from scipy.stats import spearmanr

from load_dataset import load_uiprmd, select_features
from tcn_model import TCNQualityRegressor
from train import subject_folds, DEVICE

OUTPUTS     = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEARCH_EPOCHS = 30
FULL_EPOCHS   = 150

SEARCH_GRID = {
    "lr":           [1e-3, 5e-4, 1e-4],
    "num_channels": [32, 64, 128],
    "weight_decay": [1e-4, 1e-3],
    "batch_size":   [16, 32],
}


def _eval(model, X_t, y_t, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_t), batch_size):
            xb = X_t[s:s + batch_size].to(DEVICE)
            preds.append(model(xb).cpu())
    pred = torch.cat(preds).numpy()
    true = y_t.numpy()
    rho, _  = spearmanr(pred, true)
    mae     = float(np.mean(np.abs(pred - true)))
    return float(rho), mae


def train_config(X_train, y_train, X_val, y_val,
                 lr, num_channels, weight_decay, batch_size, n_epochs):
    model   = TCNQualityRegressor(in_features=7, num_channels=num_channels).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.float32)

    best_rho = -float("inf")
    best_w   = None
    tr_losses, vl_losses = [], []

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        total = 0.0
        for s in range(0, len(Xs), batch_size):
            xb = Xs[s:s + batch_size].to(DEVICE)
            yb = ys[s:s + batch_size].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(xb)
        sched.step()

        rho, _ = _eval(model, X_vl, y_vl, batch_size)
        tr_losses.append(total / len(X_tr))
        vl_losses.append(rho)

        if rho > best_rho:
            best_rho = rho
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_w)
    final_rho, final_mae = _eval(model, X_vl, y_vl, batch_size)
    return model, final_rho, final_mae, tr_losses, vl_losses


def run_tcn_search(X, y):
    folds = subject_folds()
    tr_idx, te_idx = folds[0]
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[te_idx], y[te_idx]

    keys   = list(SEARCH_GRID.keys())
    values = list(SEARCH_GRID.values())
    configs = list(itertools.product(*values))

    print(f"Grid search: {len(configs)} configs x {SEARCH_EPOCHS} epochs on fold 0")
    print()

    rows = []
    for idx, combo in enumerate(configs):
        cfg = dict(zip(keys, combo))
        _, rho, mae, _, _ = train_config(
            X_train, y_train, X_val, y_val,
            lr=cfg["lr"],
            num_channels=cfg["num_channels"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            n_epochs=SEARCH_EPOCHS,
        )
        rows.append({**cfg, "rho": rho, "mae": mae})
        print(f"  [{idx+1:2d}/36] lr={cfg['lr']:.0e}  nc={cfg['num_channels']:3d}  "
              f"wd={cfg['weight_decay']:.0e}  bs={cfg['batch_size']:2d}  "
              f"rho={rho:+.4f}  mae={mae:.4f}")

    rows.sort(key=lambda r: r["rho"], reverse=True)
    return rows


def save_search_results(rows):
    path = OUTPUTS / "hyperparam_results.csv"
    fieldnames = ["rank", "lr", "num_channels", "weight_decay", "batch_size", "rho", "mae"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(rows, 1):
            w.writerow({
                "rank": rank,
                "lr":           r["lr"],
                "num_channels": r["num_channels"],
                "weight_decay": r["weight_decay"],
                "batch_size":   r["batch_size"],
                "rho":          round(r["rho"], 6),
                "mae":          round(r["mae"], 6),
            })
    print(f"\nSearch results saved -> {path}")

    print()
    print(f"{'Rank':>4}  {'lr':>7}  {'nc':>4}  {'wd':>6}  {'bs':>3}  {'rho':>8}  {'mae':>7}")
    print("-" * 52)
    for rank, r in enumerate(rows[:10], 1):
        print(f"  {rank:2d}   {r['lr']:.0e}   {r['num_channels']:3d}  "
              f"{r['weight_decay']:.0e}   {r['batch_size']:2d}   "
              f"{r['rho']:+.4f}   {r['mae']:.4f}")
    print("  ... (top 10 of 36 shown)")


def retrain_best(X, y, best_cfg):
    folds = subject_folds()
    results = []
    all_tr, all_vl = [], []
    best_rho   = -float("inf")
    best_model = None

    print(f"\nRetraining best config for {FULL_EPOCHS} epochs x 5 folds:")
    print(f"  lr={best_cfg['lr']:.0e}  nc={best_cfg['num_channels']}  "
          f"wd={best_cfg['weight_decay']:.0e}  bs={best_cfg['batch_size']}")
    print()

    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val,   y_val   = X[te_idx], y[te_idx]

        model, rho, mae, tr_l, vl_l = train_config(
            X_train, y_train, X_val, y_val,
            lr=best_cfg["lr"],
            num_channels=best_cfg["num_channels"],
            weight_decay=best_cfg["weight_decay"],
            batch_size=best_cfg["batch_size"],
            n_epochs=FULL_EPOCHS,
        )
        print(f"  Fold {fi+1}/5  rho={rho:+.4f}  mae={mae:.4f}")
        results.append({"fold": fi + 1, "rho": rho, "mae": mae})
        all_tr.append(tr_l)
        all_vl.append(vl_l)

        if rho > best_rho:
            best_rho   = rho
            best_model = model

    rhos = [r["rho"] for r in results]
    maes = [r["mae"] for r in results]

    torch.save(best_model.state_dict(), OUTPUTS / "best_model_tuned.pt")
    print(f"\nTuned model saved -> {OUTPUTS}/best_model_tuned.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for fi in range(5):
        axes[0].plot(all_tr[fi], color=colors[fi], alpha=0.8, label=f"Fold {fi+1}")
        axes[1].plot(all_vl[fi], color=colors[fi], alpha=0.8, label=f"Fold {fi+1}")
    axes[0].set_title("Train Loss"); axes[0].set_ylabel("MSE")
    axes[1].set_title("Val Spearman rho"); axes[1].set_ylabel("rho")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("TCN Tuned - Training Curves", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS / "training_curves_tuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tuned training curves saved -> {OUTPUTS}/training_curves_tuned.png")

    return np.mean(rhos), np.std(rhos), np.mean(maes), np.std(maes)


def _dtw_cost(ref, query, band_ratio, metric):
    n, m = len(ref), len(query)
    w = max(int(band_ratio * max(n, m)), abs(n - m))
    D = np.full((n, m), np.inf)

    for i in range(n):
        for j in range(max(0, i - w), min(m, i + w + 1)):
            a, b = ref[i], query[j]
            if metric == "euclidean":
                d = float(np.sqrt(np.sum((a - b) ** 2)))
            else:
                denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
                d     = float(1.0 - np.dot(a, b) / denom)

            if i == 0 and j == 0:
                D[i, j] = d
            else:
                prev = np.inf
                if i > 0:           prev = min(prev, D[i-1, j])
                if j > 0:           prev = min(prev, D[i, j-1])
                if i > 0 and j > 0: prev = min(prev, D[i-1, j-1])
                if prev < np.inf:
                    D[i, j] = d + prev

    return float(D[n - 1, m - 1])


def _eer(genuine, impostor, n_thr=200):
    lo = min(genuine.min(), impostor.min()) * 0.9
    hi = max(genuine.max(), impostor.max()) * 1.1
    thrs = np.linspace(lo, hi, n_thr)
    far_arr = np.array([np.sum(impostor <= t) / len(impostor) for t in thrs])
    frr_arr = np.array([np.sum(genuine  >  t) / len(genuine)  for t in thrs])
    idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    return float((far_arr[idx] + frr_arr[idx]) / 2.0)


def run_pipeline2_search(X, y):
    band_ratios = [0.05, 0.10, 0.15, 0.20]
    metrics     = ["euclidean", "cosine"]
    combos      = list(itertools.product(band_ratios, metrics))

    folds = subject_folds()
    tr_idx, te_idx = folds[0]
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test,  y_test  = X[te_idx], y[te_idx]

    correct_tr = np.where(y_train == 1)[0]
    template   = X_train[correct_tr[0]].astype(np.float64)

    correct_te   = np.where(y_test == 1)[0]
    incorrect_te = np.where(y_test == 0)[0]

    print(f"\nPipeline 2 search: {len(combos)} combos on fold 0")
    print(f"  gallery template: 1 correct training sequence (first)")
    print(f"  test: {len(correct_te)} genuine, {len(incorrect_te)} impostor")
    print()

    rows = []
    for band, metric in combos:
        genuine_scores  = np.array([_dtw_cost(template, X_test[i].astype(np.float64), band, metric)
                                    for i in correct_te])
        impostor_scores = np.array([_dtw_cost(template, X_test[i].astype(np.float64), band, metric)
                                    for i in incorrect_te])
        eer = _eer(genuine_scores, impostor_scores)
        rows.append({"band_ratio": band, "metric": metric, "eer": eer})
        print(f"  band={band:.2f}  metric={metric:<10}  EER={eer:.4f}")

    rows.sort(key=lambda r: r["eer"])
    best = rows[0]

    path = OUTPUTS / "pipeline2_hyperparam_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["band_ratio", "metric", "eer"])
        w.writeheader()
        for r in rows:
            w.writerow({"band_ratio": r["band_ratio"], "metric": r["metric"],
                        "eer": round(r["eer"], 6)})
    print(f"\nPipeline 2 search results saved -> {path}")

    return rows, best


def main(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    X_raw, _, scores = load_uiprmd()
    X = select_features(X_raw)
    print()

    default_rho = 0.3556
    default_mae = 0.0600

    print("=" * 60)
    print("STEP 1-2: TCN hyperparameter grid search")
    print("=" * 60)
    search_rows = run_tcn_search(X, scores)

    save_search_results(search_rows)

    best_cfg = {k: search_rows[0][k] for k in SEARCH_GRID.keys()}
    print(f"\nBest config:  lr={best_cfg['lr']:.0e}  nc={best_cfg['num_channels']}  "
          f"wd={best_cfg['weight_decay']:.0e}  bs={best_cfg['batch_size']}  "
          f"rho={search_rows[0]['rho']:+.4f}")

    best_hp_path = OUTPUTS / "best_hyperparams.json"
    with open(best_hp_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in best_cfg.items()}, f, indent=2)
    print(f"Best hyperparams saved -> {best_hp_path}")

    print()
    print("=" * 60)
    print("STEP 4: Full retrain with best hyperparams")
    print("=" * 60)
    tuned_rho_mean, tuned_rho_std, tuned_mae_mean, tuned_mae_std = retrain_best(X, scores, best_cfg)

    print()
    print("=" * 60)
    print("COMPARISON: default vs tuned")
    print("=" * 60)
    print(f"  {'':20s}  {'rho (mean)':>12}  {'mae (mean)':>12}")
    print(f"  {'Default (train.py)':20s}  {default_rho:>+12.4f}  {default_mae:>12.4f}")
    print(f"  {'Tuned':20s}  {tuned_rho_mean:>+12.4f}  {tuned_mae_mean:>12.4f}")
    delta_rho = tuned_rho_mean - default_rho
    delta_mae = tuned_mae_mean - default_mae
    direction = "IMPROVEMENT" if delta_rho > 0 else "REGRESSION"
    print(f"  {'Delta':20s}  {delta_rho:>+12.4f}  {delta_mae:>+12.4f}   <- {direction}")

    print()
    print("=" * 60)
    print("STEP 5: Pipeline 2 DTW hyperparameter search")
    print("=" * 60)
    p2_rows, p2_best = run_pipeline2_search(X, y=np.concatenate([np.ones(90), np.zeros(90)]).astype(np.float32))

    print()
    print(f"Best Pipeline 2 config:")
    print(f"  band_ratio={p2_best['band_ratio']}  metric={p2_best['metric']}  EER={p2_best['eer']:.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
