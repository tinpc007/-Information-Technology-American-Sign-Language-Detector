import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import gradio as gr
import subprocess


DATA_ROOT = "/mnt/d/projects/IT_model/gesture_recognition/datasets"
# ============================================================
# Configuration
# ============================================================
# Update these paths for your environment.
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
CHECKPOINT_PATH = "./runs/best.pth"
LABEL2ID_PATH = f"{DATA_ROOT}/label2id.json"
NUM_FRAMES = 16
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================
def load_label_maps(label2id_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {str(k): int(v) for k, v in label2id.items()}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def uniform_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).astype(np.int64)
    idx = np.arange(total_frames, dtype=np.int64)
    pad = np.full((num_frames - total_frames,), total_frames - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


def read_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    idx = uniform_sample_indices(total, num_frames)
    frames = vr.get_batch(idx).asnumpy()  # RGB
    return [frames[i] for i in range(frames.shape[0])]


def load_model(checkpoint_path: str, label2id_path: str, model_name: str):
    label2id, id2label = load_label_maps(label2id_path)
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={str(k): v for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.to(DEVICE)
    model.eval()
    return model, processor, label2id, id2label


MODEL, PROCESSOR, LABEL2ID, ID2LABEL = load_model(CHECKPOINT_PATH, LABEL2ID_PATH, MODEL_NAME)


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def predict_video(video_file) -> Tuple[str, List[List[str]], str]:
    if video_file is None:
        return "No video provided.", [], ""

    # Gradio may pass a filepath string or a dict-like object depending on version.
    if isinstance(video_file, str):
        video_path = video_file
    elif isinstance(video_file, dict) and "path" in video_file:
        video_path = video_file["path"]
    else:
        return f"Unsupported input format: {type(video_file)}", [], ""

    if not os.path.exists(video_path):
        return f"Video file not found: {video_path}", [], ""

    try:
        frames = read_video_frames(video_path, NUM_FRAMES)
        inputs = PROCESSOR(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        outputs = MODEL(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

        topk = min(TOP_K, probs.shape[0])
        values, indices = torch.topk(probs, k=topk)

        result_rows = []
        for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            result_rows.append([ID2LABEL[int(idx)], f"{score:.4f}"])

        pred_idx = int(indices[0].item())
        pred_label = ID2LABEL[pred_idx]
        pred_conf = float(values[0].item())

        summary = f"Prediction: {pred_label} | Confidence: {pred_conf:.4f}"
        debug = (
            f"Device: {DEVICE}\n"
            f"Frames sampled: {NUM_FRAMES}\n"
            f"Checkpoint: {CHECKPOINT_PATH}\n"
            f"Model: {MODEL_NAME}"
        )
        return summary, result_rows, debug

    except Exception as e:
        return f"Inference failed: {str(e)}", [], ""


# ============================================================
# UI
# ============================================================
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="WLASL VideoMAE Sign Recognition") as demo:
        gr.Markdown(
            """
            # WLASL Sign Recognition Web App
            Upload a video or record directly from your webcam. The app samples frames,
            runs VideoMAE inference, and returns the top predictions.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Input Video",
                    sources=["upload", "webcam"],
                    format="mp4",
                    height=360,
                )
                predict_btn = gr.Button("Recognize Sign", variant="primary")

            with gr.Column(scale=1):
                summary_output = gr.Textbox(label="Prediction Summary")
                table_output = gr.Dataframe(
                    headers=["Label", "Probability"],
                    datatype=["str", "str"],
                    row_count=TOP_K,
                    col_count=(2, "fixed"),
                    label=f"Top-{TOP_K} Predictions",
                )
                debug_output = gr.Textbox(label="Run Info", lines=4)

        examples = []
        examples_dir = Path("./examples")
        if examples_dir.exists():
            for p in sorted(examples_dir.glob("*.mp4"))[:6]:
                examples.append([str(p)])

        if examples:
            gr.Examples(examples=examples, inputs=video_input, label="Example videos")

        predict_btn.click(
            fn=predict_video,
            inputs=video_input,
            outputs=[summary_output, table_output, debug_output],
        )

        video_input.change(
            fn=predict_video,
            inputs=video_input,
            outputs=[summary_output, table_output, debug_output],
        )

    return demo


def convert_to_browser_mp4(input_path):
    output_path = input_path.replace(".mp4", "_web.mp4")

    if os.path.exists(output_path):
        return output_path

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "faststart",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path

def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
