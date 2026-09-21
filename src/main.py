import os
import sys
import io
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from cv_model import CricketClassifier
from rule_engine import CricketRuleEngine

app = FastAPI(
    title="Real-Time Cricket No-Ball Detection API (Dual CV)",
    description="Uses dual PyTorch CNNs for front-foot oversteps and waist-high beamers, backed by an ICC Law rule engine.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crease_model = None
waist_model = None
rule_engine = CricketRuleEngine()

img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_cv_models():
    global crease_model, waist_model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_backend = os.path.dirname(script_dir)
    models_dir = os.path.join(project_backend, "models")
    
    # Load Crease Model
    crease_path = os.path.join(models_dir, "crease_classifier.pth")
    crease_model = CricketClassifier()
    if os.path.exists(crease_path):
        crease_model.load_state_dict(torch.load(crease_path, map_location=device))
        crease_model.to(device)
        crease_model.eval()
        print(f"Loaded Crease Model from {crease_path}")
    else:
        print("Warning: Crease weights not found.")
        crease_model = None
        
    # Load Waist Model
    waist_path = os.path.join(models_dir, "waist_classifier.pth")
    waist_model = CricketClassifier()
    if os.path.exists(waist_path):
        waist_model.load_state_dict(torch.load(waist_path, map_location=device))
        waist_model.to(device)
        waist_model.eval()
        print(f"Loaded Waist Model from {waist_path}")
    else:
        print("Warning: Waist weights not found.")
        waist_model = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Dual-CV Cricket No-Ball Detection API is active.",
        "crease_model_loaded": crease_model is not None,
        "waist_model_loaded": waist_model is not None
    }

@app.post("/check-delivery")
async def check_delivery(
    foot_image: UploadFile = File(..., description="Image of popping crease landing"),
    waist_image: UploadFile = File(None, description="[Optional] Image of ball passing batter for beamer detection"),
    match_format: str = Form("T20", description="Format: Test, ODI, T20, 100-ball"),
    over: int = Form(0, description="Current over (0-indexed)"),
    ball_index_in_innings: int = Form(1, description="Absolute ball number in innings"),
    bouncers_in_over_before: int = Form(0, description="Bouncers bowled in current over so far"),
    is_current_ball_bouncer: bool = Form(False, description="Is this delivery a bouncer?"),
    is_beamer: bool = Form(False, description="Manual override: Is this delivery a waist-high full toss?"),
    fielders_leg_side: int = Form(0, description="Fielders on the leg side"),
    fielders_behind_square: int = Form(0, description="Fielders behind square leg"),
    fielders_outside_circle: int = Form(0, description="Fielders outside the 30-yard circle")
):
    reasons = []
    is_no_ball = False
    
    overstep_prob = 0.0
    beamer_prob = 0.0
    
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
                reasons.append(f"Front-foot overstepping detected ({overstep_prob*100:.1f}% confidence).")
    except Exception as e:
        reasons.append(f"Crease image evaluation failed: {e}")

    # 2. Run Waist Model (if provided)
    if waist_image is not None and waist_image.filename != "":
        try:
            waist_content = await waist_image.read()
            pil_waist = Image.open(io.BytesIO(waist_content)).convert("RGB")
            if waist_model is not None:
                tensor_waist = img_transform(pil_waist).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = waist_model(tensor_waist)
                    beamer_prob = F.softmax(outputs, dim=1)[0][1].item()
                    
                if beamer_prob > 0.50:
                    is_beamer = True # Auto-override manual flag
                    reasons.append(f"Waist-high full toss (beamer) detected by AI ({beamer_prob*100:.1f}% confidence).")
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
            is_beamer=is_beamer, # Will be true if manual flag OR AI detected it
            fielders_leg_side=fielders_leg_side,
            fielders_behind_square=fielders_behind_square,
            fielders_outside_circle=fielders_outside_circle
        )
        
        if rule_violation:
            is_no_ball = True
            reasons.extend([r for r in rule_reasons if "beamer" not in r.lower() or beamer_prob <= 0.50])
            if beamer_prob <= 0.50 and is_beamer:
                 reasons.append("Waist-high full toss (beamer) flagged manually.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating rule engine: {e}")

    return {
        "is_no_ball": is_no_ball,
        "ai_overstep_probability": round(overstep_prob, 4),
        "ai_beamer_probability": round(beamer_prob, 4) if waist_image is not None else None,
        "reasons": reasons
    }
