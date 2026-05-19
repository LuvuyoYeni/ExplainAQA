import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

N_SAMPLES_PER_CLASS = 90
N_FEATURES = 117
N_FRAMES   = 240

MARKER = {
    "LASI": 23, "RASI": 24, "LPSI": 25, "RPSI": 26,
    "LKNE": 28, "LTOE": 32,
    "RKNE": 34, "RANK": 36, "RTOE": 38,
}

JOINT_NAMES = [
    "left_knee", "right_knee", "left_hip", "right_hip",
    "left_ankle", "right_ankle", "pelvis_obliquity",
]


def _get_pos(seq, m):
    i = m * 3
    return seq[:, i:i+3].copy()


def _angle_at_vertex(a, b, c):
    ba  = a - b
    bc  = c - b
    den = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cos = np.clip(np.sum(ba * bc, axis=-1) / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _pelvis_obliquity(seq):
    lpsi = _get_pos(seq, MARKER["LPSI"])
    rpsi = _get_pos(seq, MARKER["RPSI"])
    diff    = rpsi - lpsi
    lateral = np.sqrt(diff[:, 0]**2 + diff[:, 2]**2) + 1e-8
    return np.degrees(np.arctan2(np.abs(diff[:, 1]), lateral)).astype(np.float32)


def load_uiprmd(data_dir=None):
    root = Path(data_dir) if data_dir else DATA_DIR

    dc = pd.read_csv(root / "Data_Correct.csv",   header=None).values
    di = pd.read_csv(root / "Data_Incorrect.csv",  header=None).values
    lc = pd.read_csv(root / "Labels_Correct.csv",  header=None).values.flatten()
    li = pd.read_csv(root / "Labels_Incorrect.csv", header=None).values.flatten()

    dc_3d = dc.reshape(N_SAMPLES_PER_CLASS, N_FEATURES, N_FRAMES).transpose(0, 2, 1)
    di_3d = di.reshape(N_SAMPLES_PER_CLASS, N_FEATURES, N_FRAMES).transpose(0, 2, 1)

    X      = np.concatenate([dc_3d, di_3d], axis=0).astype(np.float32)
    y      = np.concatenate([np.ones(N_SAMPLES_PER_CLASS),
                              np.zeros(N_SAMPLES_PER_CLASS)]).astype(np.float32)
    scores = np.concatenate([lc, li]).astype(np.float32)

    print(f"X shape:      {X.shape}   (samples, frames, features)")
    print(f"y shape:      {y.shape}   (binary: 1=correct, 0=incorrect)")
    print(f"scores shape: {scores.shape}  (continuous DTW similarity)")
    print(f"Class balance: {int(y.sum())} correct, {int((y == 0).sum())} incorrect")

    return X, y, scores


def select_features(X):
    N, T, F = X.shape
    assert F == N_FEATURES

    out = np.zeros((N, T, 7), dtype=np.float32)

    for i in range(N):
        seq = X[i].astype(np.float64)

        lasi = _get_pos(seq, MARKER["LASI"])
        rasi = _get_pos(seq, MARKER["RASI"])
        lpsi = _get_pos(seq, MARKER["LPSI"])
        rpsi = _get_pos(seq, MARKER["RPSI"])
        lkne = _get_pos(seq, MARKER["LKNE"])
        rkne = _get_pos(seq, MARKER["RKNE"])
        rank = _get_pos(seq, MARKER["RANK"])
        ltoe = _get_pos(seq, MARKER["LTOE"])
        rtoe = _get_pos(seq, MARKER["RTOE"])

        l_hjc = (lasi + lpsi) / 2.0
        r_hjc = (rasi + rpsi) / 2.0

        vert = np.zeros((T, 3), dtype=np.float64)
        vert[:, 1] = 1.0

        out[i, :, 0] = _angle_at_vertex(l_hjc, lkne, ltoe)
        out[i, :, 1] = _angle_at_vertex(r_hjc, rkne, rank)
        out[i, :, 2] = _angle_at_vertex(l_hjc + vert, l_hjc, lkne)
        out[i, :, 3] = _angle_at_vertex(r_hjc + vert, r_hjc, rkne)
        out[i, :, 4] = _angle_at_vertex(lkne, ltoe, l_hjc)
        out[i, :, 5] = _angle_at_vertex(rkne, rank, rtoe)
        out[i, :, 6] = _pelvis_obliquity(seq)

    return out


def verify(save_plot=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y, scores = load_uiprmd()
    X_final = select_features(X)

    print()
    print(f"X_final shape: {X_final.shape}  (expected: (180, 240, 7))")
    print(f"y shape:       {y.shape}")
    print(f"Class balance: {int(y.sum())} correct, {int((y == 0).sum())} incorrect")
    print()
    print(f"{'feature':22s}  {'min':>7}  {'max':>7}  {'mean':>7}")
    for j, name in enumerate(JOINT_NAMES):
        vals = X_final[:, :, j]
        print(f"{name:22s}  {vals.min():7.1f}  {vals.max():7.1f}  {vals.mean():7.1f}")

    if not save_plot:
        return X_final, y, scores

    correct_idx   = int(np.where(y == 1)[0][0])
    incorrect_idx = int(np.where(y == 0)[0][0])

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    frames = np.arange(N_FRAMES)
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        ax.plot(frames, X_final[correct_idx,   :, j], color="#1565C0", lw=1.5, label="correct")
        ax.plot(frames, X_final[incorrect_idx, :, j], color="#C62828", lw=1.5, label="incorrect", alpha=0.85)
        ax.set_ylim(0, 90 if name == "pelvis_obliquity" else 180)
        ax.set_ylabel(f"{name}\n(deg)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Frame", fontsize=10)
    fig.suptitle("UI-PRMD: one correct vs one incorrect sequence", fontsize=11)
    plt.tight_layout()

    out_path = Path(__file__).parent.parent / "outputs" / "sample_sequences.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved -> {out_path}")

    return X_final, y, scores


if __name__ == "__main__":
    verify()
