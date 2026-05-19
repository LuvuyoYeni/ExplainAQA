import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import json
from io import BytesIO
import numpy as np
import torch
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import gradio as gr

import gradio_client.utils as _gcu
_orig_j2p = _gcu._json_schema_to_python_type

def _safe_j2p(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_j2p(schema, defs)

_gcu._json_schema_to_python_type = _safe_j2p

from load_dataset import load_uiprmd, select_features, JOINT_NAMES
from tcn_model import TCNQualityRegressor
from explain import compute_attribution, check_constraints, generate_feedback
from dtw import dtw as _dtw, warp_to_reference

ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model …")
_model = TCNQualityRegressor(in_features=7).to(DEVICE)
_ckpt  = torch.load(OUTPUTS / "best_model.pt", map_location=DEVICE,
                    weights_only=True)
_model.load_state_dict(_ckpt)
_model.eval()
print(f"  ✓ Model ready  (device: {DEVICE})")

print("Loading DTW reference template …")
with open(ROOT / "data" / "aligned_reps.json") as f:
    _reps = json.load(f)
REFERENCE = np.array(_reps[0], dtype=np.float64)
print(f"  ✓ Reference shape: {REFERENCE.shape}")

_TASK_PATH = str(ROOT / "pose_landmarker.task")

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

_JOINT_TO_LM = [25, 26, 23, 24, 27, 28, 23]

_POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 31), (28, 32),
]


def calculate_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float64), \
              np.array(b, dtype=np.float64), \
              np.array(c, dtype=np.float64)
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    T, D  = seq.shape
    old_x = np.linspace(0.0, 1.0, T)
    new_x = np.linspace(0.0, 1.0, target_len)
    out   = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_x, old_x, seq[:, d].astype(np.float64))
    return out


def _fig_to_array(fig) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return img


def _make_heatmap(attr: np.ndarray, label: str,
                  pred_score: float, true_score: float = None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(12, 3))
    vmax = max(float(np.abs(attr).max()), 1e-9)
    im = ax.imshow(
        attr.T,
        aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        origin="lower", interpolation="nearest",
    )
    ax.set_yticks(range(7))
    ax.set_yticklabels(JOINT_NAMES, fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    title = f"{label}   pred = {pred_score:.4f}"
    if true_score is not None:
        title += f"   true = {true_score:.4f}"
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label="∂score/∂input")
    plt.tight_layout()
    return _fig_to_array(fig)


def _violations_html(violations: list) -> str:
    if not violations:
        return (
            "<p style='color:#2e7d32;font-weight:bold;font-size:1.1em'>"
            "✅ No constraint violations detected</p>"
        )
    items = "".join(
        f"<li style='margin:4px 0'>"
        f"<span style='color:#c62828'>❌ <b>{v['rule']}</b></span> — "
        f"{v['description']} "
        f"<span style='color:#888'>(measured {v['measured_value']}, "
        f"threshold {v['threshold']})</span></li>"
        for v in violations
    )
    return (
        f"<p style='color:#c62828;font-weight:bold'>"
        f"⚠ {len(violations)} violation(s) found</p><ul>{items}</ul>"
    )


def annotate_video(frames_bgr, lm_per_frame, resampled_seq,
                   attr, feedback, fps, frame_size, out_path):
    n_frames   = len(frames_bgr)
    W, H       = frame_size
    pred_score = feedback["predicted_score"]
    violations = feedback["constraint_violations"]

    avg_k     = (resampled_seq[:, 0] + resampled_seq[:, 1]) / 2.0
    peak_240  = int(np.argmin(avg_k))
    video_peak = int(round(peak_240 / 239.0 * max(n_frames - 1, 1)))

    viol_entries = []
    for v in violations:
        rule  = v["rule"]
        label = (f"{rule.upper().replace('_', ' ')}: "
                 f"{v['measured_value']} (limit {v['threshold']})")
        if rule in ("peak_knee_flexion", "bilateral_knee_asymmetry",
                    "descent_duration", "knee_hyperextension"):
            f_start = max(0, video_peak - 20)
            f_end   = min(n_frames - 1, video_peak + 20)
        else:
            f_start = 0
            f_end   = n_frames - 1
        viol_entries.append((label, f_start, f_end))

    if pred_score > 0.7:
        bar_color = (0, 200, 0)
    elif pred_score >= 0.5:
        bar_color = (0, 165, 255)
    else:
        bar_color = (0, 0, 255)
    bar_x, bar_y, bar_max_w, bar_h = 10, 10, 200, 22

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    for fi, frame in enumerate(frames_bgr):
        out = frame.copy()
        lms = lm_per_frame[fi] if fi < len(lm_per_frame) else None

        attr_fi = min(int(round(fi / max(n_frames - 1, 1) * 239)), 239)

        if lms is not None:
            for a_idx, b_idx in _POSE_CONNECTIONS:
                if a_idx < len(lms) and b_idx < len(lms):
                    pt_a = (int(lms[a_idx][0] * W), int(lms[a_idx][1] * H))
                    pt_b = (int(lms[b_idx][0] * W), int(lms[b_idx][1] * H))
                    cv2.line(out, pt_a, pt_b, (0, 230, 0), 2, cv2.LINE_AA)
            for lm in lms:
                pt = (int(lm[0] * W), int(lm[1] * H))
                cv2.circle(out, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, pt, 4, (0, 180, 0), 1, cv2.LINE_AA)

        if lms is not None:
            attr_row    = attr[attr_fi]
            worst_joint = int(np.argmin(attr_row))
            worst_val   = float(attr_row[worst_joint])
            if worst_val < 0:
                lm_idx = _JOINT_TO_LM[worst_joint]
                if lm_idx < len(lms):
                    pt = (int(lms[lm_idx][0] * W), int(lms[lm_idx][1] * H))
                    radius = max(6, min(40, int(abs(worst_val) * 10000)))
                    cv2.circle(out, pt, radius, (0, 0, 200), -1, cv2.LINE_AA)
                    cv2.circle(out, pt, radius, (0, 0, 255), 2, cv2.LINE_AA)

        filled_w = max(0, int(pred_score * bar_max_w))
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)
        if filled_w > 0:
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + bar_max_w, bar_y + bar_h), (180, 180, 180), 1)
        cv2.putText(out, f"Score: {pred_score:.3f}",
                    (bar_x + bar_max_w + 8, bar_y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(out, f"Frame {fi}",
                    (W - 110, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        text_y = bar_y + bar_h + 22
        for label_txt, f_start, f_end in viol_entries:
            if f_start <= fi <= f_end:
                cv2.putText(out, label_txt, (11, text_y + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, label_txt, (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (30, 30, 255), 1, cv2.LINE_AA)
                text_y += 20

        writer.write(out)

    writer.release()
    print(f"  Annotated video saved → {out_path}  "
          f"({n_frames} frames, {fps:.1f} fps, {W}x{H})")
    return str(out_path)


def process_video(video_path: str):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_TASK_PATH),
        running_mode=VisionRunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (W, H)

    angles       = []
    frames_bgr   = []
    lm_per_frame = []
    last_good    = None
    fi           = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_bgr.append(frame.copy())

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int(fi * 1000.0 / fps)
            result   = landmarker.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                def lm(i):
                    return [lms[i].x, lms[i].y]

                frame_angles = [
                    calculate_angle(lm(23), lm(25), lm(27)),
                    calculate_angle(lm(24), lm(26), lm(28)),
                    calculate_angle(lm(11), lm(23), lm(25)),
                    calculate_angle(lm(12), lm(24), lm(26)),
                    calculate_angle(lm(25), lm(27), lm(31)),
                    calculate_angle(lm(26), lm(28), lm(32)),
                    calculate_angle(lm(11), lm(23), lm(24)),
                ]
                lm_data  = [[lms[k].x, lms[k].y, lms[k].z] for k in range(33)]
                last_good = (frame_angles, lm_data)
                lm_per_frame.append(lm_data)
            else:
                frame_angles = last_good[0] if last_good is not None else [90.0] * 7
                lm_per_frame.append(last_good[1] if last_good is not None else None)

            angles.append(frame_angles)
            fi += 1

            if fi % 50 == 0:
                print(f"  frame {fi} processed …")

    cap.release()
    if not angles:
        raise ValueError("No frames could be read from the video.")
    print(f"  Done — {fi} frames extracted.")
    return np.array(angles, dtype=np.float32), frames_bgr, lm_per_frame, fps, frame_size


def _analyse_seq_video(angles, frames_bgr, lm_per_frame, fps, frame_size, label):
    path, _    = _dtw(REFERENCE, angles.astype(np.float64), band_ratio=0.1)
    aligned    = warp_to_reference(angles.astype(np.float64), path)
    resampled  = _resample(aligned, 240)

    x_in = resampled[np.newaxis]
    fb   = generate_feedback(_model, x_in, JOINT_NAMES)
    attr = compute_attribution(_model, x_in)

    heatmap   = _make_heatmap(attr, label, fb["predicted_score"])
    v_html    = _violations_html(fb["constraint_violations"])

    out_path = OUTPUTS / "annotated_output.mp4"
    annotate_video(frames_bgr, lm_per_frame, resampled,
                   attr, fb, fps, frame_size, out_path)

    return (
        round(fb["predicted_score"], 4),
        fb["feedback_text"],
        heatmap,
        v_html,
        str(out_path),
    )


def _analyse_seq(seq: np.ndarray, label: str, true_score: float = None):
    x_in = seq[np.newaxis]
    fb   = generate_feedback(_model, x_in, JOINT_NAMES)
    attr = compute_attribution(_model, x_in)

    heatmap = _make_heatmap(attr, label, fb["predicted_score"], true_score)
    v_html  = _violations_html(fb["constraint_violations"])

    return (
        round(fb["predicted_score"], 4),
        fb["feedback_text"],
        heatmap,
        v_html,
    )


def process_video_and_score(video_file):
    if video_file is None:
        return None, "No video uploaded.", None, "<p>Please upload a video first.</p>", None
    video_path = video_file.name if hasattr(video_file, "name") else str(video_file)
    print(f"\nAnalysing video: {video_path}")
    try:
        angles, frames_bgr, lm_per_frame, fps, frame_size = process_video(video_path)
        return _analyse_seq_video(angles, frames_bgr, lm_per_frame,
                                  fps, frame_size, label="Video Upload")
    except Exception as exc:
        msg = str(exc)
        return None, f"Error during analysis: {msg}", None, \
               f"<p style='color:red'>Error: {msg}</p>", None


_cache: dict = {}

def _get_dataset():
    if "X" not in _cache:
        print("Loading dataset for examples …")
        X_raw, y, scores = load_uiprmd()
        _cache["X"]      = select_features(X_raw)
        _cache["y"]      = y
        _cache["scores"] = scores
        print("  Dataset ready.")
    return _cache["X"], _cache["y"], _cache["scores"]


def load_example(label: str):
    X, y, scores = _get_dataset()
    idx          = int(np.where(y == 1)[0][0]) if label == "Correct example" \
                   else int(np.where(y == 0)[0][0])
    seq          = X[idx]
    true_score   = float(scores[idx])
    print(f"\nLoading dataset example — sample {idx} [{label}]")
    return _analyse_seq(seq, label, true_score)


def _output_panel():
    score    = gr.Number(label="Predicted Quality Score (0 – 1)", precision=4)
    feedback = gr.Textbox(label="Plain-English Feedback", lines=4,
                          interactive=False)
    heatmap  = gr.Image(label="Attribution Heatmap  (∂score / ∂joint angle)",
                        type="numpy")
    viols    = gr.HTML(label="Constraint Violations")
    return score, feedback, heatmap, viols


with gr.Blocks(title="ExplainAQA — Exercise Quality Assessment") as demo:
    gr.Markdown(
        "# ExplainAQA — Explainable Exercise Quality Assessment\n"
        "Upload a rehabilitation exercise video **or** load a dataset example "
        "to receive an AI quality score with gradient attribution and "
        "biomechanical constraint feedback."
    )

    with gr.Tabs():

        with gr.Tab("Live Analysis"):
            gr.Markdown(
                "Upload an exercise video (mp4 / avi / mov). "
                "BlazePose extracts joint angles per frame; "
                "DTW aligns the sequence to the reference template; "
                "the TCN regressor scores quality and produces an annotated video."
            )
            video_in    = gr.File(
                label="Exercise Video",
                file_types=[".mp4", ".avi", ".mov"],
                file_count="single",
            )
            analyse_btn = gr.Button("Analyse", variant="primary")

            with gr.Row():
                with gr.Column():
                    s1      = gr.Number(label="Predicted Quality Score (0 – 1)", precision=4)
                    fb1     = gr.Textbox(label="Plain-English Feedback", lines=4,
                                        interactive=False)
                    hm1     = gr.Image(label="Attribution Heatmap  (∂score / ∂joint angle)",
                                      type="numpy")
                    v1      = gr.HTML(label="Constraint Violations")
                    vid_out = gr.Video(label="Annotated Analysis Video")

            analyse_btn.click(
                fn=process_video_and_score,
                inputs=[video_in],
                outputs=[s1, fb1, hm1, v1, vid_out],
            )

        with gr.Tab("Dataset Examples"):
            gr.Markdown(
                "Load a pre-labelled sequence directly from the UI-PRMD dataset "
                "(no video upload needed). The sequence is scored and explained "
                "using the same pipeline."
            )
            example_dd = gr.Dropdown(
                choices=["Correct example", "Incorrect example"],
                value="Correct example",
                label="Select Example",
            )
            load_btn = gr.Button("Load Example", variant="primary")

            with gr.Row():
                with gr.Column():
                    s2, fb2, hm2, v2 = _output_panel()

            load_btn.click(
                fn=load_example,
                inputs=[example_dd],
                outputs=[s2, fb2, hm2, v2],
            )


if __name__ == "__main__":
    import gradio.networking as _gn
    _gn.url_ok = lambda _url: True

    print("\nStarting Gradio app on http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
