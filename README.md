# CreaseGuard Pro — Automated Cricket Officiating & No-Ball Platform

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-00a393.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Edge_Ready-5c2d91.svg)](https://onnx.ai/)

An end-to-end, real-time **Officiating-as-a-Service (OaaS)** platform and REST API designed to adjudicate illegal deliveries (no-balls) in professional and academy cricket.

**CreaseGuard Pro** combines **Dual Residual Squeeze-and-Excitation CNNs (Residual SE-CNNs)** trained on 2,897 broadcast and trajectory frames with a **Deterministic ICC/MCC Law Rule Engine** and an interactive **Third-Umpire Broadcast Web Console**.

---

## System Architecture

1. **Crease Classifier (`crease_classifier` — Residual SE-CNN)**:
   - Evaluates side-on stump/crease camera frames to detect front-foot overstepping (**MCC Law 21.5**).
   - Trained on **1,697 labeled popping crease frames** (`legal` and `no_ball`), achieving **99.71% validation accuracy**.
2. **Waist Classifier (`waist_classifier` — Residual SE-CNN)**:
   - Evaluates the batter's side-on profile to detect dangerous waist-high full tosses / beamers (**MCC Law 41.7**).
   - Trained on **1,200 labeled waist-plane trajectory frames**, achieving **96.67% validation accuracy**.
3. **Explainable AI Telemetry (Grad-CAM Activation Overlay)**:
   - Generates real-time Stage-4 spatial activation heatmaps and target reticles on every evaluated frame.
4. **Deterministic ICC/MCC Rule Engine (`rule_engine.py`)**:
   - Enforces competition-specific regulations across **T20I, IPL (2-bouncer rule), ODI (Powerplays 1–3), Test, and 100-Ball** formats, including bouncer quotas (**MCC Law 41.6**), 30-yard Powerplay circle limits (**ICC Clause 28.7**), leg-side limits (**MCC Law 28.4**), behind-square-leg limits (**MCC Law 28.4**), and **Free Hit** eligibility.
5. **CreaseGuard Pro Web Console (`frontend/`)**:
   - Web application served directly by FastAPI featuring Verified Match Presets, Dual Camera Viewports with Grad-CAM toggle, an interactive **2D Cricket Stadium & Fielding Radar (SVG)**, a Ball-by-Ball Match Audit Ledger (CSV/JSON export), a **Model Metrics Lab**, and Service Plan & API documentation.

---

## Quickstart & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Divide317/No_ball.git
   cd backend
   ```

2. **Set up Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Prepare Datasets & Train the Dual Residual SE-CNN Models:**
   ```bash
   python src/prepare_dataset.py
   python src/cv_model.py
   ```
   - Prepares and indexes the crease and waist datasets, trains both Residual SE-CNN models with `AdamW` + `OneCycleLR`, exports `.pth` and `.onnx` weights to `models/`, and writes evaluation metrics to `models/training_metrics.json`.

4. **Launch the Live Platform & API Server:**
   ```bash
   uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   - **Web Console:** Open **http://127.0.0.1:8000** in your browser.

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the CreaseGuard Pro™ SaaS Web Studio (or JSON status for API clients) |
| `POST` | `/check-delivery` | Evaluates crease & waist frames + match state; returns verdict, probabilities, Grad-CAM HUD overlays, and 6-point ICC law breakdown |
| `GET` | `/api/status` | Live system health, loaded models, ONNX readiness, and session counters |
| `GET` | `/api/metrics` | Real training & validation metrics, F1-scores, confusion matrices, and dataset provenance |
| `GET` | `/api/samples` | Returns curated real-match broadcast presets for instant 1-click testing |
| `GET` | `/api/history` | Returns the ball-by-ball session officiating ledger and summary statistics |
| `POST` | `/api/reset-history` | Clears the current session telemetry log |
