# CreaseGuard: Automated Cricket No-Ball Detection and Rule Adjudication

CreaseGuard is a computer vision and rule-checking system that evaluates cricket deliveries for no-ball infringements. It combines two convolutional neural networks for visual inspection (front-foot overstepping at the popping crease and waist-high full tosses at the batter's crease) with a deterministic rule engine covering standard ICC and MCC playing conditions across T20I, IPL, ODI, Test, and 100-ball formats.

The web interface monitors a configurable parent directory containing two subdirectories (`crease/` and `waist/`). When new camera frames are written into either subdirectory, the backend detects the updated modification timestamp, runs both vision models, generates Grad-CAM activation overlays, checks the active match context, and logs the final decision.

## System Overview

Evaluating a cricket delivery requires checking both visual evidence from side-on cameras and contextual match state:

1. **Front-Foot Popping Crease Detection (MCC Law 21.5):**
   A Residual Squeeze-and-Excitation CNN (`crease_classifier`) inspects the side-on popping crease frame to determine whether any part of the bowler's front foot landed behind the popping crease line at first impact.
2. **Waist-High Full Toss Detection (MCC Law 41.7):**
   A second Residual SE-CNN (`waist_classifier`) inspects the side-on batter frame to determine whether the ball passed above waist height on the full (a beamer).
3. **Gradient-Weighted Class Activation Mapping (Grad-CAM):**
   To make model predictions verifiable by an operator or third umpire, the backend hooks into the final convolutional stage (`stage4`) and computes a spatial activation heatmap with a peak coordinate reticle for every evaluated frame.
4. **Deterministic Rule Engine:**
   Evaluates non-visual no-ball criteria including over bouncer quotas (Law 41.6), 30-yard circle restrictions across powerplay phases (Clause 28.7), leg-side fielder limits (Law 28.4), behind-square-leg limits (Law 28.4), and format-specific Free Hit rules.

## Validation Benchmarks

Both vision models use a 4-stage Residual SE-CNN backbone (`src/model.py`) with channel dimensions `[32, 64, 128, 256]`, Squeeze-and-Excitation channel attention (`reduction=8`), concatenated global average and max pooling (`512-d`), and a two-class linear classification head. Models were trained using AdamW, a OneCycleLR learning rate schedule, and class-balanced label smoothing (`0.04`) on an 80/20 train-validation split.

| Model | Target Law | Dataset Size | Train / Val Split | Validation Accuracy | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `crease_classifier` | MCC Law 21.5 (Front-Foot Overstep) | 1,697 frames | 1,358 / 339 | 99.71% | 99.69% | 99.38% | 100.00% |
| `waist_classifier` | MCC Law 41.7 (Waist-High Full Toss) | 1,200 frames | 960 / 240 | 96.67% | 96.77% | 93.75% | 100.00% |

Trained weights are stored in `models/` in both PyTorch (`.pth`) and exported ONNX (`.onnx`) formats, along with confusion matrices in `models/training_metrics.json`.

## Directory Structure

```text
backend/
|-- frontend/
|   |-- index.html          # Web interface (Overview, Decision Console, Session Log, Benchmarks, API)
|   |-- styles.css          # Interface stylesheet
|   `-- app.js              # Folder watcher polling, SVG fielding diagram, and API client
|-- models/
|   |-- crease_classifier.pth
|   |-- crease_classifier.onnx
|   |-- waist_classifier.pth
|   |-- waist_classifier.onnx
|   `-- training_metrics.json
|-- samples/
|   |-- crease/             # Sample popping-crease frames monitored by the directory watcher
|   `-- waist/              # Sample batter waist-plane frames monitored by the directory watcher
|-- live_feed/
|   |-- crease/             # Secondary watched directory for testing live file drops
|   `-- waist/
|-- src/
|   |-- main.py             # FastAPI application, Grad-CAM hook, and folder-watcher endpoints
|   |-- model.py            # Residual SE-CNN architecture (SEBlock, ResidualSEBlock, CricketClassifier)
|   |-- train.py            # Training pipeline, data augmentation, and ONNX export
|   |-- rules.py            # ICC and MCC rule adjudication engine (NoBallRuleEngine)
|   |-- prepare_dataset.py  # Dataset indexing, waist trajectory synthesis, and sample setup
|   `-- generate_data.py    # Synthetic crease frame generator for domain augmentation
`-- requirements.txt
```

## Setup and Usage

### 1. Environment Setup

Requires Python 3.10 or newer. From the `backend` directory:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Preparation and Model Training (Optional)

Pre-trained PyTorch and ONNX checkpoints are already included in `models/`. To rebuild the dataset partitions and retrain both classifiers from scratch:

```bash
python src/prepare_dataset.py
python src/train.py
```

### 3. Running the Server

Start the FastAPI server with Uvicorn:

```bash
.\.venv\Scripts\uvicorn.exe src.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser.

### 4. Automated Subfolder Workflow

1. Navigate to the **Decision Console** tab.
2. In the **Target Folder Destination** panel, select `samples/` or `live_feed/`, enter a custom directory path containing `crease/` and `waist/` subfolders, or use **Browse Local Folder**.
3. While **Auto-Watch Active** is enabled, the frontend polls `/api/folder/status` every 1.5 seconds. Saving or copying a new image into `<folder>/crease/` or `<folder>/waist/` automatically updates the active frame selection and triggers delivery evaluation.

## REST API Reference

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web interface (or returns service status JSON when requested by an API client) |
| `GET` | `/api/status` | Returns runtime device, model load state, ONNX availability, and active watched directory |
| `GET` | `/api/folder/status` | Lists images inside `<watched_folder>/crease` and `<watched_folder>/waist` sorted newest-first |
| `POST` | `/api/folder/select` | Sets the active parent folder path to watch for `crease/` and `waist/` images |
| `GET` | `/api/folder/image/{sub}` | Serves an image file from the `crease` or `waist` subfolder |
| `POST` | `/api/folder/evaluate` | Evaluates the selected or newest images in `crease/` and `waist/` with match context parameters |
| `POST` | `/check-delivery` | Evaluates directly uploaded `foot_image` and optional `waist_image` multipart files |
| `GET` | `/api/metrics` | Returns training and validation metrics and confusion matrices from `models/training_metrics.json` |
| `GET` | `/api/history` | Returns the ball-by-ball session log and summary counts |
| `POST` | `/api/reset-history` | Clears the in-memory session log |
