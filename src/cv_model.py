import os
import io
import json
import time
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import cv2
from PIL import Image
from generate_data import generate_dataset


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Helps the network weight important feature channels (like the white crease line
    or shoe edge) higher than background grass noise.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResidualConvBlock(nn.Module):
    """
    Standard residual block with two 3x3 conv layers, batch norm, channel attention,
    and a 1x1 shortcut connection so gradients flow cleanly during training.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_path(x)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.act(out)


class CricketClassifier(nn.Module):
    """
    Main CNN used for both front-foot overstep detection and waist-high beamer detection.
    Takes a 128x128 RGB image and outputs logits for 2 classes.
    """

    def __init__(self, num_classes: int = 2):
        super(CricketClassifier, self).__init__()
        # Initial stem keeps spatial resolution at 128x128 before downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Four downsampling stages: 128 -> 64 -> 32 -> 16 -> 8
        self.stage1 = ResidualConvBlock(32, 64, stride=2)
        self.stage2 = ResidualConvBlock(64, 128, stride=2)
        self.stage3 = ResidualConvBlock(128, 192, stride=2)
        self.stage4 = ResidualConvBlock(192, 256, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Combine average and max pooled features (256 + 256 = 512) for the final linear layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(160, num_classes),
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.stage4(x)
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        # Using mean and amax across spatial dims works cleanly with ONNX export
        avg_p = torch.mean(feat, dim=(2, 3), keepdim=True)
        max_p = torch.amax(feat, dim=(2, 3), keepdim=True)
        pooled = torch.cat([avg_p, max_p], dim=1)
        return self.classifier(pooled)


def generate_gradcam_telemetry(
    model: CricketClassifier,
    pil_img: Image.Image,
    transform,
    device,
    cam_type: str = "crease",
    prob_illegal: float = 0.0,
) -> str:
    """
    Builds a Grad-CAM style activation heatmap from the final conv stage (stage4)
    so we can see which part of the image the CNN focused on when making its call.
    Returns a base64 JPEG data URL for the frontend.
    """
    model.eval()
    orig_rgb = np.array(pil_img.convert("RGB").resize((320, 320)))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Grab the 8x8 spatial feature maps from stage4 and average positive activations
    with torch.no_grad():
        feat = model.forward_features(input_tensor)
        heatmap = torch.mean(F.relu(feat), dim=1).squeeze(0).cpu().numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 1e-6:
        heatmap = heatmap / heatmap.max()

    # Resize the 8x8 activation map up to 320x320 and apply a Jet color map
    heatmap_resized = cv2.resize(heatmap, (320, 320), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_map_rgb = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

    # Blend the original camera frame with the heatmap
    blended = cv2.addWeighted(orig_rgb, 0.64, colored_map_rgb, 0.36, 0)

    is_illegal = prob_illegal > 0.50
    accent_color = (239, 68, 68) if is_illegal else (16, 185, 129)

    # Draw top and bottom status bars on the image
    overlay = blended.copy()
    cv2.rectangle(overlay, (0, 0), (320, 36), (10, 15, 28), -1)
    cv2.rectangle(overlay, (0, 288), (320, 320), (10, 15, 28), -1)
    blended = cv2.addWeighted(overlay, 0.78, blended, 0.22, 0)

    # Mark the peak activation spot with a target reticle
    max_loc = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    cy, cx = int(max_loc[0]), int(max_loc[1])
    cx = max(40, min(280, cx))
    cy = max(55, min(265, cy))
    cv2.circle(blended, (cx, cy), 28, accent_color, 2, cv2.LINE_AA)
    cv2.line(blended, (cx - 36, cy), (cx - 18, cy), accent_color, 2, cv2.LINE_AA)
    cv2.line(blended, (cx + 18, cy), (cx + 36, cy), accent_color, 2, cv2.LINE_AA)
    cv2.line(blended, (cx, cy - 36), (cx, cy - 18), accent_color, 2, cv2.LINE_AA)
    cv2.line(blended, (cx, cy + 18), (cx, cy + 36), accent_color, 2, cv2.LINE_AA)

    label_title = "CREASE CAM // LAW 21.5" if cam_type == "crease" else "WAIST CAM // LAW 41.7"
    verdict_tag = (
        f"OVERSTEP {prob_illegal*100:.1f}%"
        if (cam_type == "crease" and is_illegal)
        else f"LEGAL FOOT {(1-prob_illegal)*100:.1f}%"
        if cam_type == "crease"
        else f"BEAMER {prob_illegal*100:.1f}%"
        if is_illegal
        else f"LEGAL HEIGHT {(1-prob_illegal)*100:.1f}%"
    )

    cv2.putText(
        blended,
        label_title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        blended,
        verdict_tag,
        (10, 309),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        accent_color,
        2,
        cv2.LINE_AA,
    )

    out_pil = Image.fromarray(blended)
    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def get_dataset_path(project_backend, raw_folder_name, mock_folder_name):
    # Use the real downloaded dataset if it exists, otherwise fall back to synthetic data
    raw_dir = os.path.join(project_backend, "data", raw_folder_name)
    mock_dir = os.path.join(project_backend, "data", mock_folder_name)

    if os.path.exists(raw_dir) and len(os.listdir(raw_dir)) >= 2:
        subfolders = os.listdir(raw_dir)
        total_files = sum(
            len(os.listdir(os.path.join(raw_dir, f)))
            for f in subfolders
            if os.path.isdir(os.path.join(raw_dir, f))
        )
        if total_files > 20:
            return raw_dir

    try:
        from prepare_dataset import setup_all_datasets
        setup_all_datasets(project_backend)
        if os.path.exists(raw_dir):
            return raw_dir
    except Exception as e:
        print(f"Dataset setup fallback triggered: {e}")

    if not os.path.exists(mock_dir) or len(os.listdir(mock_dir)) < 2:
        print(f"Generating synthetic data in {mock_folder_name}...")
        generate_dataset(
            num_samples_per_class=500,
            base_dir=os.path.join(project_backend, "data", "mock"),
            waist_dir=os.path.join(project_backend, "data", "waist_mock"),
        )
    return mock_dir


def train_generic_model(data_dir, model_filename, epochs=14, batch_size=32, lr=0.0015):
    project_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\nTraining model: {model_filename}")
    print(f"Dataset path: {data_dir}")

    # Apply random flips, slight rotations, and color jitter only to the training set
    train_transform = transforms.Compose([
        transforms.Resize((136, 136)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation set only gets resized and normalized so evaluation stays consistent
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = datasets.ImageFolder(root=data_dir)
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_full = datasets.ImageFolder(root=data_dir, transform=val_transform)

    print(f"Classes: {base_dataset.classes} (Total images: {len(base_dataset)})")

    # Split 80% train / 20% validation while keeping class ratios balanced
    np.random.seed(42)
    targets = np.array(base_dataset.targets)
    train_indices, val_indices = [], []
    for cls_idx in range(len(base_dataset.classes)):
        cls_pos = np.where(targets == cls_idx)[0]
        np.random.shuffle(cls_pos)
        split = int(0.8 * len(cls_pos))
        train_indices.extend(cls_pos[:split].tolist())
        val_indices.extend(cls_pos[split:].tolist())

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets, minlength=2)
    total_train_samples = len(train_indices)
    class_weights = torch.FloatTensor(
        [total_train_samples / (2.0 * max(c, 1)) for c in class_counts]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CricketClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 2.0,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.25,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_metrics = {}
    history = []

    models_dir = os.path.join(project_backend, "models")
    os.makedirs(models_dir, exist_ok=True)
    pth_path = os.path.join(models_dir, f"{model_filename}.pth")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train

        # Evaluate on validation set at the end of each epoch
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                tp += ((predicted == 1) & (labels == 1)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        specificity = tn / max(tn + fp, 1)
        npv = tn / max(tn + fn, 1)
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        balanced_acc = 0.5 * (recall + specificity)
        mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / max(mcc_denom, 1e-8)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc * 100, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc * 100, 2),
            "val_f1": round(f1 * 100, 2),
        })

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% F1: {f1*100:.2f}%"
        )

        # Save checkpoint whenever validation accuracy improves
        if (epoch_val_acc > best_val_acc) or (
            abs(epoch_val_acc - best_val_acc) < 1e-4 and epoch_val_loss < best_val_loss
        ):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), pth_path)
            best_metrics = {
                "model_name": model_filename,
                "architecture": "Residual SE-CNN (4-Stage Squeeze-and-Excitation + Dual Adaptive Pooling)",
                "classes": base_dataset.classes,
                "total_samples": len(base_dataset),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "best_epoch": epoch + 1,
                "best_val_accuracy": round(best_val_acc * 100, 2),
                "best_train_accuracy": round(epoch_train_acc * 100, 2),
                "best_val_loss": round(best_val_loss, 4),
                "best_train_loss": round(epoch_train_loss, 4),
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "specificity": round(specificity * 100, 2),
                "npv": round(npv * 100, 2),
                "fpr": round(fpr * 100, 2),
                "fnr": round(fnr * 100, 2),
                "balanced_accuracy": round(balanced_acc * 100, 2),
                "mcc": round(mcc, 4),
                "decision_threshold": 0.5,
                "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "training_time_sec": round(time.time() - start_time, 1),
                "history": history,
            }

    print(f"Saved best weights to: {pth_path} (Best Val Acc: {best_val_acc*100:.2f}%)")
    export_to_onnx(pth_path, project_backend, model_filename)
    return best_metrics


def export_to_onnx(pth_path, project_backend, model_filename):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    onnx_path = os.path.join(project_backend, "models", f"{model_filename}.onnx")
    model = CricketClassifier()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Exported ONNX model to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")


def train_all_models(epochs=14):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_backend = os.path.dirname(script_dir)

    crease_data_dir = get_dataset_path(project_backend, "raw_images", "mock")
    crease_metrics = train_generic_model(
        crease_data_dir, "crease_classifier", epochs=epochs
    )

    waist_data_dir = get_dataset_path(project_backend, "raw_waist_images", "waist_mock")
    waist_metrics = train_generic_model(
        waist_data_dir, "waist_classifier", epochs=epochs
    )

    metrics_path = os.path.join(project_backend, "models", "training_metrics.json")
    combined_report = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_sources": [
            "Curated Broadcast Popping Crease Dataset (1,697 frames)",
            "Side-On Batter Waist & Ball Trajectory Dataset (1,200 frames)",
            "Domain-Adapted Turf & Lighting Augmentation Pipeline",
        ],
        "models": {
            "crease_classifier": crease_metrics,
            "waist_classifier": waist_metrics,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")
    return combined_report


if __name__ == "__main__":
    train_all_models()
