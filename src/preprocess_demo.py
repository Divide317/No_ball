"""
preprocess_demo.py - Demonstrates the preprocessing pipeline in preprocessing.py.

Run from the backend folder:
    python src/preprocess_demo.py                 # quality report + stage images
    python src/preprocess_demo.py --compare-models # also compares CNN predictions
                                                   # on original vs enhanced frames

Outputs go to backend/preprocessing_output/ (a new folder; nothing else is touched):
    <image>_stages.jpg   side-by-side panel of every preprocessing stage
    report.json          quality metrics, issues and (optionally) model outputs
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from preprocessing import (  # noqa: E402
    load_image,
    assess_quality,
    enhance_frame,
    to_model_tensor,
)

PROJECT_BACKEND = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_BACKEND)
OUTPUT_DIR = os.path.join(PROJECT_BACKEND, "preprocessing_output")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_sample_folder():
    for base in (WORKSPACE_ROOT, PROJECT_BACKEND):
        p = os.path.join(base, "samples")
        if os.path.isdir(p):
            return p
    return None


def build_stage_panel(stages, tile=200):
    """Puts every stage next to each other with its name printed underneath."""
    tiles = []
    for name, rgb in stages:
        t = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile + 28, tile, 3), 18, dtype=np.uint8)
        canvas[:tile] = t
        cv2.putText(canvas, name.upper(), (8, tile + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(canvas)
    sep = np.full((tile + 28, 4, 3), 60, dtype=np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [sep, t]
    return np.hstack(row)


def load_models():
    """Loads both trained CNNs (needs PyTorch, which the project venv already has)."""
    import torch
    from torchvision import transforms
    from cv_model import CricketClassifier

    models = {}
    for key in ("crease", "waist"):
        path = os.path.join(PROJECT_BACKEND, "models", f"{key}_classifier.pth")
        if os.path.exists(path):
            m = CricketClassifier()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            models[key] = m

    # Exactly the transform used in main.py for inference
    reference_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch, models, reference_tf


def illegal_probability(torch, model, tensor_np, cam):
    import torch.nn.functional as F
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(tensor_np)), dim=1)[0]
    # Class order: crease -> ['legal', 'no_ball']; waist -> ['beamer', 'legal_waist']
    return float(probs[1] if cam == "crease" else probs[0])


def main():
    parser = argparse.ArgumentParser(description="CreaseGuard Pro preprocessing demo")
    parser.add_argument("--folder", help="Folder containing crease/ and waist/ subfolders")
    parser.add_argument("--compare-models", action="store_true",
                        help="Run both CNNs on original vs enhanced frames")
    args = parser.parse_args()

    folder = args.folder or find_sample_folder()
    if not folder or not os.path.isdir(folder):
        print("No sample folder found. Pass one with --folder.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch = models = reference_tf = None
    if args.compare_models:
        try:
            torch, models, reference_tf = load_models()
        except Exception as e:
            print(f"Model comparison unavailable ({e}); continuing without it.")
            args.compare_models = False

    results = []
    print(f"\nPreprocessing frames from: {folder}\n")
    header = f"{'camera':7} {'file':42} {'bright':>7} {'contr':>6} {'sharp':>8}  issues"
    print(header)
    print("-" * len(header))

    for cam in ("crease", "waist"):
        sub = os.path.join(folder, cam)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if os.path.splitext(fname)[1].lower() not in VALID_EXTS:
                continue
            path = os.path.join(sub, fname)

            img = load_image(path)
            report = assess_quality(img)
            enhanced, stages = enhance_frame(img, report)
            after = assess_quality(enhanced)

            tensor = to_model_tensor(img)
            stem = os.path.splitext(fname)[0]
            panel_path = os.path.join(OUTPUT_DIR, f"{cam}_{stem}_stages.jpg")
            Image.fromarray(build_stage_panel(stages)).save(panel_path, quality=92)

            entry = {
                "camera": cam,
                "file": fname,
                "quality_before": report.to_dict(),
                "quality_after_enhancement": after.to_dict(),
                "model_tensor": {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": round(float(tensor.min()), 4),
                    "max": round(float(tensor.max()), 4),
                    "per_channel_mean": [round(float(v), 4) for v in tensor[0].mean(axis=(1, 2))],
                },
                "stage_panel": os.path.relpath(panel_path, PROJECT_BACKEND),
            }

            if args.compare_models and cam in models:
                ref = reference_tf(img).unsqueeze(0).numpy()
                entry["numpy_vs_torchvision_max_abs_diff"] = round(float(np.abs(ref - tensor).max()), 6)
                p_orig = illegal_probability(torch, models[cam], ref, cam)
                p_enh = illegal_probability(torch, models[cam], reference_tf(enhanced).unsqueeze(0).numpy(), cam)
                entry["illegal_probability_original"] = round(p_orig, 4)
                entry["illegal_probability_enhanced"] = round(p_enh, 4)

            results.append(entry)
            issues = ", ".join(report.issues) or "ok"
            print(f"{cam:7} {fname[:42]:42} {report.brightness:7.1f} {report.contrast:6.1f} "
                  f"{report.sharpness:8.1f}  {issues}")
            if "illegal_probability_original" in entry:
                print(f"{'':7} -> P(illegal) original {entry['illegal_probability_original']:.3f}"
                      f" | enhanced {entry['illegal_probability_enhanced']:.3f}"
                      f" | numpy/torchvision diff {entry['numpy_vs_torchvision_max_abs_diff']}")

    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"source_folder": folder, "frames": results}, f, indent=2)

    print(f"\nProcessed {len(results)} frames.")
    print(f"Stage panels and report.json written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
