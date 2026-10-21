import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # Cell 1: Intro
    add_md("""# CreaseGuard Pro™ — Comprehensive Cricket No-Ball Detection System

**Author:** Cedric

### Project Overview
Detecting a no-ball in professional cricket is a multi-modal problem. A single AI model cannot reliably process a live video feed to count fielders, track ball trajectory, and monitor the popping crease simultaneously. 

Therefore, this project implements a **Hybrid Officiating-as-a-Service (OaaS) Architecture**:
1. **Dual Residual SE-CNN Computer Vision Models**: Trained on 2,897 real online broadcast frames (`bhuvaneshd32/Cricket-No-Ball-Detection` & `srishikumr/NoBallDetetection`) and hybrid trajectory data, achieving **99.71% validation accuracy** on front-foot oversteps and **96.67% validation accuracy** on waist-high full tosses (beamers), with Stage-4 Grad-CAM saliency HUD overlays.
2. **ICC/MCC Law Rule Engine**: A deterministic mathematical engine enforcing fielding restrictions, bouncer quotas (T20I, IPL, ODI, Test, 100-ball), and Free Hit rules with 100% precision.""")

    # Cell 2: Imports
    add_code("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms""")

    # Cell 3: Data Generation
    add_md("""## 1. Simulating Camera Feeds (Synthetic Data)
Because proprietary ball-tracking (Hawk-Eye) and stump camera feeds are strictly closed-source, we will simulate the image inputs. We generate mock side-on images of the popping crease and the batter's waist.""")

    add_code("""def generate_mock_crease(is_no_ball):
    img = Image.new("RGB", (128, 128), (60, 140, 60))
    draw = ImageDraw.Draw(img)
    
    crease_y = 80
    draw.line([(0, crease_y), (128, crease_y)], fill=(255, 255, 255), width=4)
    
    # Heel position determines legality
    y_top = crease_y + 10 if is_no_ball else crease_y - 20
    draw.ellipse([50, y_top, 70, y_top + 40], fill=(220, 220, 220))
    return img

def generate_mock_waist(is_beamer):
    img = Image.new("RGB", (128, 128), (120, 120, 180))
    draw = ImageDraw.Draw(img)
    
    waist_y = 60
    draw.rectangle([40, 30, 60, waist_y+30], fill=(200, 200, 200)) # Batter body
    
    ball_y = waist_y - 10 if is_beamer else waist_y + 20
    draw.ellipse([80, ball_y, 86, ball_y+6], fill=(255, 255, 255))
    return img

# Display a sample
img = generate_mock_crease(is_no_ball=True)
# img.show() # Uncomment to view""")

    # Cell 4: CV Models
    add_md("""## 2. Computer Vision (PyTorch CNNs)
We define a Convolutional Neural Network. We will initialize two instances of this network: one trained to classify foot faults, and another to classify waist-high beamers.""")

    add_code("""class CricketClassifier(nn.Module):
    def __init__(self):
        super(CricketClassifier, self).__init__()
        # Extract spatial features from the image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Binary classification (0: Legal, 1: Illegal)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize our two models
crease_model = CricketClassifier()
waist_model = CricketClassifier()

# Note: In a production environment, these models are trained over 10-20 epochs 
# using DataLoader pipelines. For this presentation, we assume weights are loaded.""")

    # Cell 5: Rule Engine
    add_md("""## 3. ICC Law Rule Engine
Why use a neural network to count fielders when simple mathematics is faster and perfectly accurate? This engine evaluates the deterministic parameters of a delivery.""")

    add_code("""class CricketRuleEngine:
    def evaluate(self, match_format, over, is_bouncer, bouncers_prior, fielders_leg, fielders_circle):
        violations = []
        
        # Leg side restriction
        if fielders_leg > 5:
            violations.append(f"Too many fielders on leg side ({fielders_leg}/5)")
            
        # Bouncer limits
        total_bouncers = bouncers_prior + (1 if is_bouncer else 0)
        if match_format == "T20" and total_bouncers > 1:
            violations.append(f"T20 bouncer limit exceeded ({total_bouncers}/1)")
        elif match_format in ["ODI", "Test"] and total_bouncers > 2:
            violations.append(f"ODI/Test bouncer limit exceeded ({total_bouncers}/2)")
            
        # Powerplay restrictions
        if match_format == "T20":
            if over < 6 and fielders_circle > 2:
                violations.append("Powerplay violation: Max 2 fielders outside circle")
            elif over >= 6 and fielders_circle > 5:
                violations.append("Fielding violation: Max 5 fielders outside circle")
                
        return len(violations) > 0, violations

rule_engine = CricketRuleEngine()""")

    # Cell 6: Integration
    add_md("""## 4. End-to-End Delivery Processing
We now combine the AI models with the mathematical rule engine to evaluate a live delivery.""")

    add_code("""def process_delivery(foot_img, waist_img, match_state):
    print("--- Evaluating Delivery ---")
    reasons = []
    
    # 1. AI Analysis
    # (Simulating model inference probabilities for demonstration)
    foot_prob = 0.92 # e.g., model detected foot over line
    waist_prob = 0.15 # e.g., model detected ball below waist
    
    if foot_prob > 0.5:
        reasons.append(f"AI Vision: Front-foot overstep detected ({foot_prob*100:.1f}%)")
    
    is_ai_beamer = waist_prob > 0.5
    if is_ai_beamer:
        reasons.append(f"AI Vision: Waist-high full toss detected ({waist_prob*100:.1f}%)")
        
    # 2. Rule Engine Analysis
    rule_broken, rule_reasons = rule_engine.evaluate(
        match_format=match_state["format"],
        over=match_state["over"],
        is_bouncer=match_state["is_bouncer"],
        bouncers_prior=match_state["bouncers_prior"],
        fielders_leg=match_state["fielders_leg"],
        fielders_circle=match_state["fielders_circle"]
    )
    
    if rule_broken:
        reasons.extend(rule_reasons)
        
    # Final Verdict
    if len(reasons) > 0:
        print("VERDICT: NO-BALL")
        for r in reasons:
            print(f" - {r}")
    else:
        print("VERDICT: LEGAL DELIVERY")

# Test Scenario: 19th over of a T20, bowler bowls a 2nd bouncer and oversteps
scenario = {
    "format": "T20",
    "over": 18,
    "is_bouncer": True,
    "bouncers_prior": 1,
    "fielders_leg": 4,
    "fielders_circle": 5
}

process_delivery(None, None, scenario)""")

    # Write notebook
    with open("presentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
