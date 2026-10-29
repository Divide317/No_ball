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

## Comparison with Existing Approaches

Most prior automated no-ball systems and academic prototypes focus exclusively on binary image classification of the bowler's front foot at the popping crease. In real matches, however, an umpire must adjudicate several distinct laws simultaneously, verify why a model flagged a marginal frame, and ingest frames continuously from camera feeds without manual file uploads.

CreaseGuard addresses the main limitations of earlier approaches across five areas:

1. **Dual-Camera Coverage Instead of Single-Crease Classification:**
   Earlier computer vision implementations only evaluate front-foot overstepping (Law 21.5) and ignore height-based infringements. CreaseGuard runs two separate neural networks in parallel: one for the popping crease landing (`crease_classifier`) and a second for waist-high full tosses at the batter's plane (`waist_classifier`, Law 41.7).
2. **Channel-Attention Residual Architecture vs. Shallow CNNs / Uncalibrated Transfer Learning:**
   Standard shallow CNNs and generic pre-trained backbones often struggle with motion blur, turf discoloration, and stadium shadow gradients around the white crease line. CreaseGuard introduces Squeeze-and-Excitation (SE) channel attention inside each residual stage to weight feature channels corresponding to the crease edge, shoe heel, and ball trajectory while suppressing background grass noise, alongside dual average-and-max global pooling.
3. **Hybrid Vision and Deterministic Rule Adjudication:**
   Prior vision-only tools cannot detect tactical no-balls such as fielding circle violations, leg-side fielder limits, or exceeded bouncer quotas, nor do they account for format differences (for example, the two-bouncer-per-over allowance in IPL versus one in T20I, or three powerplay blocks in ODI). CreaseGuard couples neural inference with a stateful rule engine so that all six ICC and MCC criteria are checked in a single pass.
4. **Verifiable Spatial Attribution (Grad-CAM) vs. Black-Box Probabilities:**
   A raw softmax probability is insufficient when a foot landing is within a few millimeters of the popping crease. CreaseGuard extracts Stage-4 spatial gradients on every inference call to overlay a heatmap and peak activation coordinate directly onto the camera frame, showing whether the network focused on the shoe heel and crease line or on a background artifact.
5. **Automated Subfolder Ingestion vs. Manual Single-File Scripts:**
   Instead of requiring an operator to run a command-line script or manually upload individual files after each ball, CreaseGuard monitors a parent folder with dedicated `crease/` and `waist/` subdirectories, automatically detecting and evaluating new frames as they are written to disk.

| Approach | Scope of Laws Checked | Model Explainability | Match Context Awareness | Frame Ingestion Workflow |
| :--- | :--- | :--- | :--- | :--- |
| Manual on-field umpiring | All laws, but subject to human split-second error at the crease | None | High, tracked manually by umpires and scorers | Visual observation only |
| Proprietary stadium Hawk-Eye / TV replay | Front-foot replay and ball tracking | Trajectory graphics, no neural attribution | Requires manual third-umpire review per ball | Multi-camera broadcast truck hardware |
| Prior single-camera CNN prototypes | Front-foot overstep only (Law 21.5) | Black-box class label or probability | None (ignores format, bouncers, and fielding) | Manual single-image upload or CLI script |
| **CreaseGuard (Proposed)** | **Dual-CV (Laws 21.5 & 41.7) + 4 contextual ICC/MCC rules** | **Stage-4 Grad-CAM heatmap and peak reticle** | **Format-aware engine (T20I, IPL, ODI, Test, 100-Ball)** | **Automated directory watcher (`crease/` & `waist/` subfolders)** |

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
