import os
import sys
import io
import json
import time
import uuid
from typing import Optional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

from cv_model import CricketClassifier, generate_gradcam_telemetry
from rule_engine import CricketRuleEngine
from generate_data import create_synthetic_crease_image, create_synthetic_waist_image

app = FastAPI(
    title="CreaseGuard Pro — AI Cricket Officiating & No-Ball SaaS API",
    description="Dual Residual SE-CNN Computer Vision models trained on real cricket broadcast datasets combined with a deterministic ICC/MCC Law Rule Engine.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crease_model: Optional[CricketClassifier] = None
waist_model: Optional[CricketClassifier] = None
rule_engine = CricketRuleEngine()
startup_time = time.time()
delivery_history = []

img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_BACKEND = os.path.dirname(SCRIPT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_BACKEND, "frontend")
MODELS_DIR = os.path.join(PROJECT_BACKEND, "models")
SAMPLES_DIR = os.path.join(PROJECT_BACKEND, "data", "samples")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


SAMPLE_PRESETS = [
    {
        "id": "real_overstep",
        "title": "Real Broadcast: Front-Foot Overstep",
        "subtitle": "ICC Law 21.5 • Real TV Stump/Crease Camera Frame",
        "badge": "NO-BALL (AI VISION)",
        "badge_type": "danger",
        "foot_file": "real_noball_1.jpg",
        "waist_file": "waist_legal_1.jpg",
        "context": {
            "match_format": "T20",
            "over": 17,
            "ball_index_in_innings": 104,
            "bouncers_in_over_before": 0,
            "is_current_ball_bouncer": False,
            "is_beamer": False,
            "fielders_leg_side": 4,
            "fielders_behind_square": 1,
            "fielders_outside_circle": 5,
            "bowler_name": "M. Starc",
            "batter_name": "V. Kohli",
        },
    },
    {
        "id": "real_legal",
        "title": "Real Broadcast: Fair Delivery Behind Line",
        "subtitle": "ICC Law 21.5 • Heel Grounded Behind Popping Crease",
        "badge": "LEGAL DELIVERY",
        "badge_type": "success",
        "foot_file": "real_legal_1.jpg",
        "waist_file": "waist_legal_1.jpg",
        "context": {
            "match_format": "T20",
            "over": 3,
            "ball_index_in_innings": 21,
            "bouncers_in_over_before": 0,
            "is_current_ball_bouncer": False,
            "is_beamer": False,
            "fielders_leg_side": 3,
            "fielders_behind_square": 1,
            "fielders_outside_circle": 2,
            "bowler_name": "J. Bumrah",
            "batter_name": "J. Buttler",
        },
    },
    {
        "id": "real_tight_noball",
        "title": "TV Third-Umpire Review: Close Margin Overstep",
        "subtitle": "ICC Law 21.5 • Millimeter Crease Line Evaluation",
        "badge": "NO-BALL (CLOSE CALL)",
        "badge_type": "danger",
        "foot_file": "real_noball_2.jpg",
        "waist_file": "waist_legal_1.jpg",
        "context": {
            "match_format": "ODI",
            "over": 46,
            "ball_index_in_innings": 278,
            "bouncers_in_over_before": 1,
            "is_current_ball_bouncer": False,
            "is_beamer": False,
            "fielders_leg_side": 4,
            "fielders_behind_square": 2,
            "fielders_outside_circle": 5,
            "bowler_name": "P. Cummins",
            "batter_name": "R. Sharma",
        },
    },
    {
        "id": "waist_beamer",
        "title": "Dangerous Delivery: Waist-High Beamer",
        "subtitle": "MCC Law 41.7 • Full Toss Above Batter Waist Plane",
        "badge": "NO-BALL (BEAMER AI)",
        "badge_type": "danger",
        "foot_file": "real_legal_2.jpg",
        "waist_file": "waist_beamer_1.jpg",
        "context": {
            "match_format": "IPL",
            "over": 19,
            "ball_index_in_innings": 116,
            "bouncers_in_over_before": 1,
            "is_current_ball_bouncer": False,
            "is_beamer": False,
            "fielders_leg_side": 4,
            "fielders_behind_square": 1,
            "fielders_outside_circle": 5,
            "bowler_name": "K. Rabada",
            "batter_name": "H. Klaasen",
        },
    },
    {
        "id": "powerplay_breach",
        "title": "Tactical Fault: Powerplay Circle Violation",
        "subtitle": "ICC Clause 28.7 • 3 Fielders Outside 30-Yard Circle in Over 4",
        "badge": "NO-BALL (FIELDING RULE)",
        "badge_type": "warning",
        "foot_file": "real_legal_1.jpg",
        "waist_file": "waist_legal_1.jpg",
        "context": {
            "match_format": "T20",
            "over": 3,
            "ball_index_in_innings": 22,
            "bouncers_in_over_before": 0,
            "is_current_ball_bouncer": False,
            "is_beamer": False,
            "fielders_leg_side": 4,
            "fielders_behind_square": 1,
            "fielders_outside_circle": 3,
            "bowler_name": "T. Boult",
            "batter_name": "S. Gill",
        },
    },
    {
        "id": "bouncer_exceeded",
        "title": "Bowler Quota: 2nd Bouncer in T20I Over",
        "subtitle": "MCC Law 41.6 • Exceeds 1 Short-Pitched Limit in T20I",
        "badge": "NO-BALL (BOUNCER LIMIT)",
        "badge_type": "warning",
        "foot_file": "real_legal_2.jpg",
        "waist_file": "waist_legal_1.jpg",
        "context": {
            "match_format": "T20",
            "over": 14,
            "ball_index_in_innings": 87,
            "bouncers_in_over_before": 1,
            "is_current_ball_bouncer": True,
            "is_beamer": False,
            "fielders_leg_side": 4,
            "fielders_behind_square": 1,
            "fielders_outside_circle": 5,
            "bowler_name": "M. Wood",
            "batter_name": "S. Yadav",
        },
    },
]


@app.on_event("startup")
def load_cv_models():
    global crease_model, waist_model

    # Load Crease Model
    crease_path = os.path.join(MODELS_DIR, "crease_classifier.pth")
    crease_model = CricketClassifier()
    if os.path.exists(crease_path):
        try:
            crease_model.load_state_dict(torch.load(crease_path, map_location=device))
            crease_model.to(device)
            crease_model.eval()
            print(f"Loaded Crease Model from {crease_path}")
        except Exception as e:
            print(f"Warning: Failed to load Crease weights ({e}).")
            crease_model = None
    else:
        print("Warning: Crease weights not found.")
        crease_model = None

    # Load Waist Model
    waist_path = os.path.join(MODELS_DIR, "waist_classifier.pth")
    waist_model = CricketClassifier()
    if os.path.exists(waist_path):
        try:
            waist_model.load_state_dict(torch.load(waist_path, map_location=device))
            waist_model.to(device)
            waist_model.eval()
            print(f"Loaded Waist Model from {waist_path}")
        except Exception as e:
            print(f"Warning: Failed to load Waist weights ({e}).")
            waist_model = None
    else:
        print("Warning: Waist weights not found.")
        waist_model = None


@app.get("/")
def read_root(request: Request):
    frontend_index = os.path.join(FRONTEND_DIR, "index.html")
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {
        "status": "online",
        "service": "CreaseGuard Pro — Dual-CV Cricket No-Ball Detection SaaS",
        "version": "3.0.0",
        "message": "Dual-CV Cricket No-Ball Detection API is active.",
        "crease_model_loaded": crease_model is not None,
        "waist_model_loaded": waist_model is not None,
    }


@app.get("/app")
def serve_web_app():
    frontend_index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/api/status")
def get_api_status():
    return {
        "status": "online",
        "version": "3.0.0",
        "device": str(device),
        "uptime_sec": round(time.time() - startup_time, 1),
        "crease_model_loaded": crease_model is not None,
        "waist_model_loaded": waist_model is not None,
        "crease_onnx_ready": os.path.exists(os.path.join(MODELS_DIR, "crease_classifier.onnx")),
        "waist_onnx_ready": os.path.exists(os.path.join(MODELS_DIR, "waist_classifier.onnx")),
        "deliveries_analyzed_session": len(delivery_history),
    }


@app.get("/api/metrics")
def get_training_metrics():
    metrics_path = os.path.join(MODELS_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {
        "updated_at": None,
        "dataset_sources": ["Synthetic Fallback"],
        "models": {},
    }


@app.get("/api/samples")
def get_sample_presets():
    return {"samples": SAMPLE_PRESETS}


@app.get("/api/samples/{sample_id}/image/{cam_type}")
def get_sample_image(sample_id: str, cam_type: str):
    preset = next((p for p in SAMPLE_PRESETS if p["id"] == sample_id), None)
    if not preset:
        raise HTTPException(status_code=404, detail="Sample preset not found")

    filename = preset["foot_file"] if cam_type == "foot" else preset["waist_file"]
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")

    # Fallback on-the-fly generation if sample file missing
    is_illegal = "noball" in filename or "beamer" in filename
    img = (
        create_synthetic_crease_image(is_no_ball=is_illegal)
        if cam_type == "foot"
        else create_synthetic_waist_image(is_beamer=is_illegal)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return JSONResponse(status_code=404, content={"detail": "Sample image missing"})


@app.get("/api/history")
def get_delivery_history():
    total = len(delivery_history)
    no_balls = sum(1 for d in delivery_history if d["is_no_ball"])
    oversteps = sum(1 for d in delivery_history if d["ai_overstep_probability"] > 0.5)
    beamers = sum(
        1
        for d in delivery_history
        if (d["ai_beamer_probability"] or 0) > 0.5 or d.get("manual_beamer")
    )
    return {
        "summary": {
            "total_deliveries": total,
            "legal_deliveries": total - no_balls,
            "no_balls": no_balls,
            "no_ball_rate_pct": round((no_balls / total) * 100, 1) if total > 0 else 0.0,
            "ai_oversteps": oversteps,
            "beamers_flagged": beamers,
        },
        "deliveries": list(reversed(delivery_history[-50:])),
    }


@app.post("/api/reset-history")
def reset_delivery_history():
    delivery_history.clear()
    return {"status": "cleared", "total_deliveries": 0}


@app.post("/check-delivery")
async def check_delivery(
    foot_image: UploadFile = File(..., description="Image of popping crease landing"),
    waist_image: UploadFile = File(None, description="[Optional] Image of ball passing batter for beamer detection"),
    match_format: str = Form("T20", description="Format: Test, ODI, T20, IPL, 100-ball"),
    over: int = Form(0, description="Current over (0-indexed)"),
    ball_index_in_innings: int = Form(1, description="Absolute ball number in innings"),
    bouncers_in_over_before: int = Form(0, description="Bouncers bowled in current over so far"),
    is_current_ball_bouncer: bool = Form(False, description="Is this delivery a bouncer?"),
    is_beamer: bool = Form(False, description="Manual override: Is this delivery a waist-high full toss?"),
    fielders_leg_side: int = Form(0, description="Fielders on the leg side"),
    fielders_behind_square: int = Form(0, description="Fielders behind square leg"),
    fielders_outside_circle: int = Form(0, description="Fielders outside the 30-yard circle"),
    bowler_name: str = Form("Bowler", description="Bowler name for telemetry log"),
    batter_name: str = Form("Batter", description="Batter name for telemetry log"),
):
    t0 = time.perf_counter()
    reasons = []
    is_no_ball = False

    overstep_prob = 0.0
    beamer_prob = 0.0
    has_waist_img = False
    foot_telemetry_b64 = None
    waist_telemetry_b64 = None
    manual_beamer_flag = bool(is_beamer)

    # 1. Run Crease Model
    try:
        foot_content = await foot_image.read()
        pil_foot = Image.open(io.BytesIO(foot_content)).convert("RGB")
        if crease_model is not None:
            tensor_foot = img_transform(pil_foot).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = crease_model(tensor_foot)
                overstep_prob = F.softmax(outputs, dim=1)[0][1].item()

            if overstep_prob > 0.50:
                is_no_ball = True
                reasons.append(
                    f"Front-foot overstepping detected by AI Vision ({overstep_prob*100:.1f}% confidence, MCC Law 21.5)."
                )

            foot_telemetry_b64 = generate_gradcam_telemetry(
                crease_model,
                pil_foot,
                img_transform,
                device,
                cam_type="crease",
                prob_illegal=overstep_prob,
            )
    except Exception as e:
        reasons.append(f"Crease image evaluation failed: {e}")

    # 2. Run Waist Model (if provided)
    if waist_image is not None and getattr(waist_image, "filename", "") != "":
        try:
            waist_content = await waist_image.read()
            if len(waist_content) > 0:
                has_waist_img = True
                pil_waist = Image.open(io.BytesIO(waist_content)).convert("RGB")
                if waist_model is not None:
                    tensor_waist = img_transform(pil_waist).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = waist_model(tensor_waist)
                        # In ImageFolder(['beamer', 'legal_waist']), 'beamer' is class index 0
                        beamer_prob = F.softmax(outputs, dim=1)[0][0].item()

                    if beamer_prob > 0.50:
                        is_beamer = True  # Auto-override manual flag
                        reasons.append(
                            f"Waist-high full toss (beamer) detected by AI Vision ({beamer_prob*100:.1f}% confidence, MCC Law 41.7)."
                        )

                    waist_telemetry_b64 = generate_gradcam_telemetry(
                        waist_model,
                        pil_waist,
                        img_transform,
                        device,
                        cam_type="waist",
                        prob_illegal=beamer_prob,
                    )
        except Exception as e:
            reasons.append(f"Waist image evaluation failed: {e}")

    # 3. Run Law-Based Rule Engine
    try:
        rule_violation, rule_reasons = rule_engine.check_delivery(
            match_format=match_format,
            over=over,
            ball_index_in_innings=ball_index_in_innings,
            bouncers_in_over_before=bouncers_in_over_before,
            is_current_ball_bouncer=is_current_ball_bouncer,
            is_beamer=is_beamer,
            fielders_leg_side=fielders_leg_side,
            fielders_behind_square=fielders_behind_square,
            fielders_outside_circle=fielders_outside_circle,
        )

        if rule_violation:
            is_no_ball = True
            reasons.extend(
                [r for r in rule_reasons if "beamer" not in r.lower() or beamer_prob <= 0.50]
            )
            if beamer_prob <= 0.50 and is_beamer:
                reasons.append("Waist-high full toss (beamer) flagged manually (MCC Law 41.7).")

        rule_details = rule_engine.check_delivery_detailed(
            match_format=match_format,
            over=over,
            ball_index_in_innings=ball_index_in_innings,
            bouncers_in_over_before=bouncers_in_over_before,
            is_current_ball_bouncer=is_current_ball_bouncer,
            is_beamer=is_beamer,
            fielders_leg_side=fielders_leg_side,
            fielders_behind_square=fielders_behind_square,
            fielders_outside_circle=fielders_outside_circle,
            overstep_prob=overstep_prob,
            beamer_prob=beamer_prob if has_waist_img else (1.0 if is_beamer else 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating rule engine: {e}")

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    verdict = "NO-BALL" if is_no_ball else "LEGAL DELIVERY"

    record = {
        "delivery_id": str(uuid.uuid4())[:8].upper(),
        "timestamp": time.strftime("%H:%M:%S"),
        "match_format": match_format,
        "over_ball": f"{over}.{((ball_index_in_innings - 1) % 6) + 1}",
        "bowler_name": bowler_name,
        "batter_name": batter_name,
        "is_no_ball": is_no_ball,
        "verdict": verdict,
        "free_hit_awarded": rule_details["free_hit_awarded"],
        "ai_overstep_probability": round(overstep_prob, 4),
        "ai_beamer_probability": round(beamer_prob, 4) if has_waist_img else None,
        "manual_beamer": manual_beamer_flag,
        "reasons": reasons,
        "inference_time_ms": inference_ms,
    }
    delivery_history.append(record)

    return {
        "delivery_id": record["delivery_id"],
        "timestamp": record["timestamp"],
        "is_no_ball": is_no_ball,
        "verdict": verdict,
        "free_hit_awarded": rule_details["free_hit_awarded"],
        "ai_overstep_probability": round(overstep_prob, 4),
        "ai_beamer_probability": round(beamer_prob, 4) if has_waist_img else None,
        "inference_time_ms": inference_ms,
        "reasons": reasons,
        "rule_details": rule_details,
        "telemetry": {
            "foot_gradcam": foot_telemetry_b64,
            "waist_gradcam": waist_telemetry_b64,
        },
    }
