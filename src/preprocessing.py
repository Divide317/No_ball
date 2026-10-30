"""
preprocessing.py - Standalone image preprocessing pipeline for CreaseGuard.

This module is NOT imported by main.py or cv_model.py, so the live server and the
trained models behave exactly as before. It exists to make every preprocessing
step explicit, inspectable and measurable (for demonstration / reporting).

Pipeline stages
---------------
1. load_image()          Decode, fix camera EXIF rotation, force 3-channel RGB.
2. assess_quality()      Measure brightness, contrast and sharpness (blur) and flag
                         frames that are too dark / washed out / blurry to trust.
3. enhance_frame()       Optional classical CV clean-up:
                           a. Gamma correction for under/over-exposed frames
                           b. Non-local-means denoising (broadcast compression noise)
                           c. CLAHE contrast boost on the L channel (LAB space) so the
                              white crease line and shoe edge stand out from the turf
                           d. Unsharp masking to recover edges lost to motion blur
4. to_model_tensor()     The exact preprocessing the CNNs use at inference time
                         (Resize 128x128 -> scale to [0,1] -> ImageNet mean/std
                         normalisation -> HWC to CHW), re-implemented in NumPy so
                         each arithmetic step can be shown and verified.

Only OpenCV, NumPy and Pillow are required (all already in requirements.txt).
"""

from dataclasses import dataclass, asdict, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Same constants used by main.py's img_transform and cv_model.py's val_transform
MODEL_INPUT_SIZE: Tuple[int, int] = (128, 128)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Quality thresholds (tuned for 8-bit broadcast-style frames)
DARK_THRESHOLD = 60.0        # mean luminance below this -> under-exposed
BRIGHT_THRESHOLD = 200.0     # mean luminance above this -> over-exposed
LOW_CONTRAST_THRESHOLD = 25  # luminance std-dev below this -> flat / hazy
BLUR_THRESHOLD = 100.0       # variance of Laplacian below this -> blurry
MIN_SIDE_PX = 64             # smaller frames carry too little detail


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_image(path_or_file) -> Image.Image:
    """
    Opens an image and normalises its format:
      - applies EXIF orientation (phone/camera frames are often stored rotated)
      - converts palette / RGBA / greyscale / CMYK images to 3-channel RGB
    """
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# 2. Quality assessment
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    width: int
    height: int
    brightness: float          # mean luminance, 0-255
    contrast: float            # std-dev of luminance
    sharpness: float           # variance of the Laplacian (higher = sharper)
    issues: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self):
        d = asdict(self)
        d["is_usable"] = self.is_usable
        return d


def assess_quality(pil_img: Image.Image) -> QualityReport:
    """
    Computes simple, explainable image-quality metrics:
      brightness - mean grey level
      contrast   - standard deviation of grey levels
      sharpness  - variance of the Laplacian (2nd derivative). A blurred image
                   has few strong edges, so its Laplacian response is flat and
                   the variance is low.
    """
    rgb = np.asarray(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues = []
    if min(h, w) < MIN_SIDE_PX:
        issues.append(f"low resolution ({w}x{h})")
    if brightness < DARK_THRESHOLD:
        issues.append("under-exposed")
    elif brightness > BRIGHT_THRESHOLD:
        issues.append("over-exposed")
    if contrast < LOW_CONTRAST_THRESHOLD:
        issues.append("low contrast")
    if sharpness < BLUR_THRESHOLD:
        issues.append("blurry")

    return QualityReport(
        width=w,
        height=h,
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2),
        issues=issues,
    )


# --------------------------------------------------------------------------- #
# 3. Enhancement (classical computer vision)
# --------------------------------------------------------------------------- #
def gamma_correct(rgb: np.ndarray, brightness: float) -> np.ndarray:
    """
    Brightens dark frames (gamma < 1) or darkens washed-out frames (gamma > 1)
    using a lookup table: out = 255 * (in / 255) ** gamma.
    Gamma is chosen so the mean brightness moves toward mid-grey (~128).
    """
    if DARK_THRESHOLD <= brightness <= BRIGHT_THRESHOLD:
        return rgb
    mean_norm = np.clip(brightness / 255.0, 1e-3, 0.999)
    gamma = float(np.clip(np.log(0.5) / np.log(mean_norm), 0.4, 2.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(rgb, lut)


def denoise(rgb: np.ndarray, strength: int = 5) -> np.ndarray:
    """Non-local-means denoising: removes JPEG/broadcast grain while keeping edges."""
    return cv2.fastNlMeansDenoisingColored(rgb, None, strength, strength, 7, 21)


def apply_clahe(rgb: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation on the L (lightness) channel
    of LAB colour space. Working on L only boosts local contrast - e.g. the white
    popping crease against green turf - without shifting the colours.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)


def unsharp_mask(rgb: np.ndarray, amount: float = 0.6, sigma: float = 1.2) -> np.ndarray:
    """Sharpens edges: out = in + amount * (in - GaussianBlur(in))."""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)


def enhance_frame(pil_img: Image.Image, report: QualityReport = None):
    """
    Runs the enhancement chain and returns (final_image, list_of_(name, image)
    intermediate stages) so every step can be visualised.
    """
    if report is None:
        report = assess_quality(pil_img)

    stages = []
    rgb = np.asarray(pil_img).copy()
    stages.append(("original", rgb))

    rgb = gamma_correct(rgb, report.brightness)
    stages.append(("gamma", rgb))

    rgb = denoise(rgb)
    stages.append(("denoised", rgb))

    rgb = apply_clahe(rgb)
    stages.append(("clahe", rgb))

    rgb = unsharp_mask(rgb)
    stages.append(("sharpened", rgb))

    return Image.fromarray(rgb), stages


# --------------------------------------------------------------------------- #
# 4. Model input tensor (mirrors torchvision transforms in main.py)
# --------------------------------------------------------------------------- #
def to_model_tensor(pil_img: Image.Image) -> np.ndarray:
    """
    NumPy re-implementation of:
        transforms.Resize((128, 128))
        transforms.ToTensor()                 -> uint8 [0,255] to float [0,1], HWC->CHW
        transforms.Normalize(mean, std)       -> (x - mean) / std per channel
    Returns a float32 array of shape (1, 3, 128, 128), ready for the CNN.
    """
    resized = pil_img.resize(MODEL_INPUT_SIZE, Image.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0      # HWC, [0, 1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD                  # per-channel z-score
    x = np.transpose(x, (2, 0, 1))                          # CHW
    return np.expand_dims(x, 0).astype(np.float32)          # NCHW


def preprocess(path_or_file, enhance: bool = True):
    """
    Convenience wrapper: load -> assess -> (enhance) -> tensor.
    Returns a dict with the tensor, the quality report and the processed image.
    """
    img = load_image(path_or_file)
    report = assess_quality(img)
    processed, stages = enhance_frame(img, report) if enhance else (img, [("original", np.asarray(img))])
    return {
        "tensor": to_model_tensor(processed),
        "quality": report,
        "image": processed,
        "stages": stages,
    }
