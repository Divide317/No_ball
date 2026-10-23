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
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

from cv_model import CricketClassifier, generate_gradcam_telemetry
from rule_engine import CricketRuleEngine

# Initialize FastAPI server
app = FastAPI(
    title="CreaseGuard Pro - Cricket No-Ball Detection API",
    description="Combines dual PyTorch CNNs for crease and waist checks with an ICC/MCC rule engine and automated folder watching.",
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

# Keep an in-memory log of deliveries evaluated during the current session
delivery_history = []

# Same image preprocessing pipeline used during validation
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_BACKEND = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_BACKEND)
FRONTEND_DIR = os.path.join(PROJECT_BACKEND, "frontend")
MODELS_DIR = os.path.join(PROJECT_BACKEND, "models")
SAMPLES_DIR = os.path.join(PROJECT_BACKEND, "data", "samples")

# Default watched directory containing 'crease' and 'waist' subfolders
DEFAULT_WATCH_DIR = os.path.join(WORKSPACE_ROOT, "samples")
if not os.path.exists(DEFAULT_WATCH_DIR):
    DEFAULT_WATCH_DIR = os.path.join(PROJECT_BACKEND, "samples")

watch_state = {
    "folder_path": DEFAULT_WATCH_DIR,
}

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class FolderSelectRequest(BaseModel):
    folder_path: str


def resolve_watch_folder(raw_path: str) -> str:
    """
    Resolves user-supplied folder paths (either absolute paths or relative folder
    names like 'samples' or 'live_feed') to an existing directory on disk.
    """
    cleaned = raw_path.strip().strip('"').strip("'")
    if os.path.isabs(cleaned) and os.path.exists(cleaned):
        return os.path.abspath(cleaned)

    # Check relative to workspace root and backend root
    for base in [WORKSPACE_ROOT, PROJECT_BACKEND]:
        candidate = os.path.abspath(os.path.join(base, cleaned))
        if os.path.exists(candidate):
            return candidate

    # If it doesn't exist yet, create it under workspace root
    target = (
        os.path.abspath(cleaned)
        if os.path.isabs(cleaned)
        else os.path.abspath(os.path.join(WORKSPACE_ROOT, cleaned))
    )
    os.makedirs(os.path.join(target, "crease"), exist_ok=True)
    os.makedirs(os.path.join(target, "waist"), exist_ok=True)
    return target


def scan_watched_subfolders(folder_path: str):
    """
    Scans the 'crease' and 'waist' subfolders inside the target directory.
    Sorts images by modification time (newest first) so newly added files are
    picked up automatically as soon as they land in the folder.
    """
    crease_dir = os.path.join(folder_path, "crease")
    waist_dir = os.path.join(folder_path, "waist")
    os.makedirs(crease_dir, exist_ok=True)
    os.makedirs(waist_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def list_images(sub_dir):
        items = []
        if not os.path.exists(sub_dir):
            return items
        for fname in os.listdir(sub_dir):
            ext = os.path.splitext(fname)[1].lower()
            full_p = os.path.join(sub_dir, fname)
            if ext in valid_exts and os.path.isfile(full_p):
                mtime = os.path.getmtime(full_p)
                items.append({"name": fname, "mtime": round(mtime, 2), "size": os.path.getsize(full_p)})
        # Sort newest modified/added file first, then alphabetically
        items.sort(key=lambda x: (-x["mtime"], x["name"]))
        return items

    crease_files = list_images(crease_dir)
    waist_files = list_images(waist_dir)

    latest_crease = crease_files[0]["name"] if crease_files else None
    latest_waist = waist_files[0]["name"] if waist_files else None

    sig_c = f"{latest_crease}:{crease_files[0]['mtime']}" if crease_files else "none"
    sig_w = f"{latest_waist}:{waist_files[0]['mtime']}" if waist_files else "none"
    signature = f"{folder_path}|{len(crease_files)}|{sig_c}|{len(waist_files)}|{sig_w}"

    return {
        "folder_path": folder_path,
        "crease_subfolder": crease_dir,
        "waist_subfolder": waist_dir,
        "crease_files": crease_files,
        "waist_files": waist_files,
        "latest_crease": latest_crease,
        "latest_waist": latest_waist,
        "signature": signature,
    }


def run_delivery_pipeline(
    pil_foot: Image.Image,
    pil_waist: Optional[Image.Image],
    match_format: str,
    over: int,
    ball_index_in_innings: int,
    bouncers_in_over_before: int,
    is_current_ball_bouncer: bool,
    is_beamer: bool,
    fielders_leg_side: int,
    fielders_behind_square: int,
    fielders_outside_circle: int,
    bowler_name: str = "Bowler",
    batter_name: str = "Batter",
    foot_source_name: str = "crease.jpg",
    waist_source_name: Optional[str] = None,
):
    """
    Core adjudication pipeline shared by both folder-watcher ingestion
    and direct multipart uploads.
    """
    t0 = time.perf_counter()
    reasons = []
    is_no_ball = False

    overstep_prob = 0.0
    beamer_prob = 0.0
    has_waist_img = pil_waist is not None
    foot_telemetry_b64 = None
    waist_telemetry_b64 = None
    manual_beamer_flag = bool(is_beamer)

    # Step 1: Evaluate front-foot landing with the Crease CNN
    try:
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

    # Step 2: Evaluate waist-high full toss if a waist frame is present
    if pil_waist is not None:
        try:
            if waist_model is not None:
                tensor_waist = img_transform(pil_waist).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = waist_model(tensor_waist)
                    # In ImageFolder(['beamer', 'legal_waist']), 'beamer' is class index 0
                    beamer_prob = F.softmax(outputs, dim=1)[0][0].item()

                if beamer_prob > 0.50:
                    is_beamer = True
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

    # Step 3: Run deterministic ICC/MCC rule checks
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

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    verdict = "NO-BALL" if is_no_ball else "LEGAL DELIVERY"

    record = {
        "delivery_id": str(uuid.uuid4())[:8].upper(),
        "timestamp": time.strftime("%H:%M:%S"),
        "match_format": match_format,
        "over_ball": f"{over}.{((ball_index_in_innings - 1) % 6) + 1}",
        "bowler_name": bowler_name,
        "batter_name": batter_name,
        "foot_file": foot_source_name,
        "waist_file": waist_source_name,
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
        "foot_file": foot_source_name,
        "waist_file": waist_source_name,
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


@app.on_event("startup")
def load_cv_models():
    """Load both trained PyTorch checkpoints into memory when the server starts."""
    global crease_model, waist_model

    crease_path = os.path.join(MODELS_DIR, "crease_classifier.pth")
    crease_model = CricketClassifier()
    if os.path.exists(crease_path):
        try:
            crease_model.load_state_dict(torch.load(crease_path, map_location=device))
            crease_model.to(device)
            crease_model.eval()
            print(f"Loaded Crease Model from {crease_path}")
        except Exception as e:
            print(f"Failed to load Crease weights ({e}).")
            crease_model = None
    else:
        print("Warning: Crease weights not found.")
        crease_model = None

    waist_path = os.path.join(MODELS_DIR, "waist_classifier.pth")
    waist_model = CricketClassifier()
    if os.path.exists(waist_path):
        try:
            waist_model.load_state_dict(torch.load(waist_path, map_location=device))
            waist_model.to(device)
            waist_model.eval()
            print(f"Loaded Waist Model from {waist_path}")
        except Exception as e:
            print(f"Failed to load Waist weights ({e}).")
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
        "service": "CreaseGuard Pro - Dual-CV Cricket No-Ball Detection API",
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
        "watched_folder": watch_state["folder_path"],
    }


@app.get("/api/folder/status")
def get_watched_folder_status():
    """Returns the current state of the watched folder and its 'crease' and 'waist' subfolders."""
    return scan_watched_subfolders(watch_state["folder_path"])


@app.post("/api/folder/select")
def set_watched_folder(req: FolderSelectRequest):
    """Updates the target folder destination to monitor for 'crease' and 'waist' images."""
    resolved = resolve_watch_folder(req.folder_path)
    watch_state["folder_path"] = resolved
    return scan_watched_subfolders(resolved)


@app.get("/api/folder/image/{sub}")
def get_watched_folder_image(sub: str, file: str):
    """Serves an image file directly from the watched 'crease' or 'waist' subfolder."""
    if sub not in ["crease", "waist"]:
        raise HTTPException(status_code=400, detail="Subfolder must be 'crease' or 'waist'")
    safe_file = os.path.basename(file)
    img_path = os.path.join(watch_state["folder_path"], sub, safe_file)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found in watched subfolder")
    return FileResponse(img_path)


@app.post("/api/folder/evaluate")
def evaluate_from_watched_folder(
    crease_file: Optional[str] = Form(None),
    waist_file: Optional[str] = Form(None),
    match_format: str = Form("T20"),
    over: int = Form(3),
    ball_index_in_innings: int = Form(21),
    bouncers_in_over_before: int = Form(0),
    is_current_ball_bouncer: bool = Form(False),
    is_beamer: bool = Form(False),
    fielders_leg_side: int = Form(3),
    fielders_behind_square: int = Form(1),
    fielders_outside_circle: int = Form(2),
    bowler_name: str = Form("J. Bumrah"),
    batter_name: str = Form("J. Buttler"),
):
    """
    Automatically picks up the latest (or specified) images from the watched
    <folder>/crease and <folder>/waist subfolders and runs the full detection pipeline.
    """
    status = scan_watched_subfolders(watch_state["folder_path"])
    c_name = crease_file or status["latest_crease"]
    w_name = waist_file if waist_file is not None else status["latest_waist"]

    if not c_name:
        raise HTTPException(
            status_code=400,
            detail=f"No images found inside {status['crease_subfolder']}",
        )

    c_path = os.path.join(status["crease_subfolder"], os.path.basename(c_name))
    if not os.path.exists(c_path):
        raise HTTPException(status_code=404, detail=f"Crease image not found: {c_name}")

    pil_foot = Image.open(c_path).convert("RGB")
    pil_waist = None
    if w_name and w_name != "NONE":
        w_path = os.path.join(status["waist_subfolder"], os.path.basename(w_name))
        if os.path.exists(w_path):
            pil_waist = Image.open(w_path).convert("RGB")

    return run_delivery_pipeline(
        pil_foot=pil_foot,
        pil_waist=pil_waist,
        match_format=match_format,
        over=over,
        ball_index_in_innings=ball_index_in_innings,
        bouncers_in_over_before=bouncers_in_over_before,
        is_current_ball_bouncer=is_current_ball_bouncer,
        is_beamer=is_beamer,
        fielders_leg_side=fielders_leg_side,
        fielders_behind_square=fielders_behind_square,
        fielders_outside_circle=fielders_outside_circle,
        bowler_name=bowler_name,
        batter_name=batter_name,
        foot_source_name=c_name,
        waist_source_name=w_name if pil_waist else None,
    )


@app.get("/api/metrics")
def get_training_metrics():
    metrics_path = os.path.join(MODELS_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "updated_at": None,
        "dataset_sources": ["Curated Broadcast Dataset"],
        "models": {},
    }


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
    bowler_name: str = Form("Bowler", description="Bowler name for match log"),
    batter_name: str = Form("Batter", description="Batter name for match log"),
):
    foot_content = await foot_image.read()
    pil_foot = Image.open(io.BytesIO(foot_content)).convert("RGB")

    pil_waist = None
    waist_name = None
    if waist_image is not None and getattr(waist_image, "filename", "") != "":
        waist_content = await waist_image.read()
        if len(waist_content) > 0:
            pil_waist = Image.open(io.BytesIO(waist_content)).convert("RGB")
            waist_name = waist_image.filename

    return run_delivery_pipeline(
        pil_foot=pil_foot,
        pil_waist=pil_waist,
        match_format=match_format,
        over=over,
        ball_index_in_innings=ball_index_in_innings,
        bouncers_in_over_before=bouncers_in_over_before,
        is_current_ball_bouncer=is_current_ball_bouncer,
        is_beamer=is_beamer,
        fielders_leg_side=fielders_leg_side,
        fielders_behind_square=fielders_behind_square,
        fielders_outside_circle=fielders_outside_circle,
        bowler_name=bowler_name,
        batter_name=batter_name,
        foot_source_name=foot_image.filename or "crease.jpg",
        waist_source_name=waist_name,
    )
