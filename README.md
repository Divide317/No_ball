# Hybrid Cricket No-Ball Detection System 🏏

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-00a393.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

An end-to-end, real-time backend API designed to detect illegal deliveries (no-balls) in professional cricket. 

This project solves the multi-modal challenges of cricket umpiring by combining **Dual Computer Vision Models (PyTorch CNNs)** for spatial recognition with a **Deterministic Rule Engine** for strict mathematical compliance with ICC/MCC laws.

---

## 🏗️ System Architecture

A single "God Model" cannot reliably track ball trajectories, count fielders, and monitor crease landings simultaneously. Instead, this system uses a modular **Hybrid Architecture**:

1. **Crease Classifier (AI)**: A CNN that evaluates side-on camera frames of the popping crease to detect front-foot overstepping.
2. **Waist Classifier (AI)**: A secondary CNN that evaluates the batter's side-on profile to detect waist-high full tosses (beamers).
3. **ICC Law Rule Engine (Logic)**: A Python engine that enforces format-specific rules (T20, ODI, Test, 100-ball), tracking bouncer limits, Powerplay circle restrictions, and leg-side fielding limits.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd backend
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training the Models (Synthetic Data)

Because multi-camera Hawk-Eye and stump camera datasets are closed-source and proprietary, this repository includes a robust **Synthetic Data Generator**.

1. **Generate Mock Training Data:**
   ```bash
   python src/generate_data.py
   ```
   *Generates 2,000 images (foot landings and waist-high trajectories).*

2. **Train the Dual PyTorch CNNs:**
   ```bash
   python src/cv_model.py
   ```
   *Trains both the `crease_classifier` and `waist_classifier` and exports the `.pth` and `.onnx` weights to the `models/` directory.*

---

## ⚡ Running the Live API

Start the FastAPI backend server:

```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Navigate to **http://127.0.0.1:8000/docs** in your browser to access the interactive Swagger UI dashboard.

### API Usage
The `POST /check-delivery` endpoint requires:
* `foot_image` (File): Frame of the popping crease.
* `waist_image` (File, Optional): Frame of the ball passing the batter.
* Match Context: Form data including `match_format`, `over`, `bouncers_in_over_before`, and field placements.

The API returns a consolidated verdict detailing whether the delivery is legal, or listing the specific AI/Rule-based reasons for flagging a no-ball.

---

**Disclaimer:** This is an academic/demonstration project simulating professional sports tech methodologies.
