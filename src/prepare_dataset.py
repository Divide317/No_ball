import os
import json
import shutil
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from generate_data import create_synthetic_crease_image


def verify_and_index_crease_dataset(project_backend: str):
    """
    Scans data/raw_images/{legal,no_ball}, verifies that all image files are valid
    RGB JPEGs, and normalizes their filenames for the training loader.
    """
    raw_dir = os.path.join(project_backend, "data", "raw_images")
    legal_dir = os.path.join(raw_dir, "legal")
    noball_dir = os.path.join(raw_dir, "no_ball")
    os.makedirs(legal_dir, exist_ok=True)
    os.makedirs(noball_dir, exist_ok=True)

    counts = {"legal": 0, "no_ball": 0}

    for label, folder, prefix in [
        ("legal", legal_dir, "crease_legal"),
        ("no_ball", noball_dir, "crease_noball"),
    ]:
        files = sorted(
            [
                f
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ]
        )
        idx = 1
        for fname in files:
            old_path = os.path.join(folder, fname)
            new_name = f"{prefix}_{idx:04d}.jpg"
            new_path = os.path.join(folder, new_name)
            try:
                if old_path != new_path and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                counts[label] += 1
                idx += 1
            except Exception:
                pass

    return counts


def render_broadcast_waist_frame(is_beamer: bool, real_bg_path=None, size=(160, 160)):
    """
    Renders a side-on delivery frame showing the batter at the popping crease
    and the ball approaching either above waist height (beamer) or below waist height.
    """
    w, h = size
    if real_bg_path and os.path.exists(real_bg_path):
        try:
            bg = Image.open(real_bg_path).convert("RGB").resize(size)
            bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.65, 0.9))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=1.8))
        except Exception:
            bg = None
    else:
        bg = None

    if bg is None:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        sky_col = np.array(
            [random.randint(70, 135), random.randint(95, 155), random.randint(130, 195)]
        )
        turf_col = np.array(
            [random.randint(45, 85), random.randint(105, 155), random.randint(45, 85)]
        )
        pitch_col = np.array(
            [random.randint(175, 210), random.randint(155, 185), random.randint(120, 150)]
        )
        horizon = random.randint(int(h * 0.56), int(h * 0.66))
        arr[:horizon, :] = sky_col
        arr[horizon:, :] = turf_col
        arr[int(h * 0.68) :, :] = pitch_col
        noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)

    draw = ImageDraw.Draw(bg, "RGBA")

    # Popping crease line on the pitch
    crease_y = random.randint(int(h * 0.76), int(h * 0.84))
    draw.line([(0, crease_y), (w, crease_y)], fill=(245, 245, 245, 210), width=3)

    # Batter standing at the crease
    batter_x = random.randint(int(w * 0.18), int(w * 0.30))
    waist_y = random.randint(int(h * 0.45), int(h * 0.54))
    head_y = waist_y - random.randint(26, 34)
    foot_y = crease_y + random.randint(-2, 6)

    jersey_colors = [
        (30, 90, 210, 255),
        (220, 180, 30, 255),
        (20, 140, 80, 255),
        (235, 235, 240, 255),
        (190, 40, 50, 255),
    ]
    jersey = random.choice(jersey_colors)
    pad_color = (230, 230, 235, 255)

    draw.rounded_rectangle(
        [batter_x - 7, waist_y, batter_x + 9, foot_y], radius=4, fill=pad_color
    )
    draw.rounded_rectangle(
        [batter_x - 9, head_y + 10, batter_x + 11, waist_y + 4], radius=5, fill=jersey
    )
    draw.ellipse(
        [batter_x - 7, head_y - 4, batter_x + 9, head_y + 11],
        fill=(25, 45, 95, 255),
    )
    bat_bottom_y = waist_y + random.randint(8, 26)
    draw.line(
        [(batter_x + 6, waist_y - 2), (batter_x + 18, bat_bottom_y)],
        fill=(205, 165, 105, 255),
        width=5,
    )

    # Dashed waist reference line
    for x_seg in range(0, w, 12):
        draw.line(
            [(x_seg, waist_y), (min(x_seg + 6, w), waist_y)],
            fill=(255, 70, 70, 175),
            width=2,
        )

    ball_x = random.randint(batter_x + 18, int(w * 0.78))
    ball_radius = random.randint(4, 6)
    is_white_ball = random.random() > 0.45
    ball_color = (250, 250, 252, 255) if is_white_ball else (200, 25, 35, 255)
    trail_color = (255, 255, 255, 95) if is_white_ball else (255, 80, 80, 95)

    if is_beamer:
        # Full toss passing above waist level without bouncing
        ball_y = random.randint(max(12, head_y - 4), waist_y - 7)
        release_y = random.randint(int(h * 0.22), int(h * 0.36))
        draw.line(
            [(w - 5, release_y), (ball_x, ball_y)],
            fill=trail_color,
            width=3,
        )
    else:
        # Legal delivery below waist height
        ball_y = random.randint(waist_y + 8, crease_y - 4)
        bounce_x = min(w - 12, ball_x + random.randint(15, 45))
        bounce_y = crease_y - random.randint(0, 6)
        draw.line(
            [(w - 5, int(h * 0.28)), (bounce_x, bounce_y), (ball_x, ball_y)],
            fill=trail_color,
            width=3,
        )

    draw.ellipse(
        [
            ball_x - ball_radius,
            ball_y - ball_radius,
            ball_x + ball_radius,
            ball_y + ball_radius,
        ],
        fill=ball_color,
        outline=(30, 30, 30, 180),
    )

    if random.random() > 0.55:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.25, 0.7)))

    return bg.convert("RGB")


def build_waist_dataset(project_backend: str, num_samples_per_class: int = 600):
    """
    Ensures data/raw_waist_images/{legal_waist,beamer} is populated with
    labeled side-on waist trajectory frames.
    """
    raw_crease_legal = os.path.join(project_backend, "data", "raw_images", "legal")
    waist_dir = os.path.join(project_backend, "data", "raw_waist_images")
    legal_waist_dir = os.path.join(waist_dir, "legal_waist")
    beamer_dir = os.path.join(waist_dir, "beamer")
    os.makedirs(legal_waist_dir, exist_ok=True)
    os.makedirs(beamer_dir, exist_ok=True)

    bg_files = []
    if os.path.exists(raw_crease_legal):
        bg_files = [
            os.path.join(raw_crease_legal, f)
            for f in os.listdir(raw_crease_legal)
            if f.endswith(".jpg")
        ]

    random.seed(101)
    for cls_name, is_beamer in [("legal_waist", False), ("beamer", True)]:
        target_dir = beamer_dir if is_beamer else legal_waist_dir
        for i in range(num_samples_per_class):
            out_path = os.path.join(target_dir, f"{cls_name}_{i:04d}.jpg")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 512:
                continue

            use_real_bg = len(bg_files) > 0 and (i % 2 == 0)
            img = render_broadcast_waist_frame(
                is_beamer=is_beamer,
                real_bg_path=random.choice(bg_files) if use_real_bg else None,
                size=(160, 160),
            )
            img.save(out_path, format="JPEG", quality=92)

    return {
        "legal_waist": len(os.listdir(legal_waist_dir)),
        "beamer": len(os.listdir(beamer_dir)),
    }


def augment_crease_with_synthetic_frames(project_backend: str, count_per_class: int = 200):
    """
    Supplements the match crease images with synthetic turf frames so the model
    handles both broadcast camera angles and synthetic test cases.
    """
    raw_dir = os.path.join(project_backend, "data", "raw_images")
    legal_dir = os.path.join(raw_dir, "legal")
    noball_dir = os.path.join(raw_dir, "no_ball")
    os.makedirs(legal_dir, exist_ok=True)
    os.makedirs(noball_dir, exist_ok=True)

    if len(os.listdir(legal_dir)) < 700:
        for i in range(count_per_class):
            p_legal = os.path.join(legal_dir, f"turf_aug_legal_{i:04d}.jpg")
            if not os.path.exists(p_legal):
                create_synthetic_crease_image(is_no_ball=False, size=(160, 160)).save(
                    p_legal, format="JPEG", quality=92
                )
    if len(os.listdir(noball_dir)) < 700:
        for i in range(count_per_class):
            p_nb = os.path.join(noball_dir, f"turf_aug_noball_{i:04d}.jpg")
            if not os.path.exists(p_nb):
                create_synthetic_crease_image(is_no_ball=True, size=(160, 160)).save(
                    p_nb, format="JPEG", quality=92
                )


def prepare_curated_samples(project_backend: str):
    """
    Copies verified test images into data/samples/ and samples/ for live demos.
    """
    samples_dir = os.path.join(project_backend, "data", "samples")
    backend_samples = os.path.join(project_backend, "samples")
    workspace_samples = os.path.join(os.path.dirname(project_backend), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(backend_samples, exist_ok=True)
    os.makedirs(workspace_samples, exist_ok=True)

    demo_mappings = [
        ("real_noball_1.jpg", "1_front_foot_noball_overstep.jpg"),
        ("real_noball_2.jpg", "2_front_foot_noball_close_margin.jpg"),
        ("real_legal_1.jpg", "3_legal_delivery_behind_crease.jpg"),
        ("real_legal_2.jpg", "4_legal_delivery_fair_landing.jpg"),
        ("waist_beamer_1.jpg", "5_waist_high_beamer_noball.jpg"),
        ("waist_legal_1.jpg", "6_waist_legal_height_delivery.jpg"),
    ]
    for src_name, demo_name in demo_mappings:
        src_p = os.path.join(samples_dir, src_name)
        if os.path.exists(src_p):
            shutil.copyfile(src_p, os.path.join(backend_samples, demo_name))
            shutil.copyfile(src_p, os.path.join(workspace_samples, demo_name))


def setup_all_datasets(project_backend: str = None):
    if project_backend is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_backend = os.path.dirname(script_dir)

    augment_crease_with_synthetic_frames(project_backend, count_per_class=200)
    crease_counts = verify_and_index_crease_dataset(project_backend)
    waist_counts = build_waist_dataset(project_backend, num_samples_per_class=600)
    prepare_curated_samples(project_backend)

    summary = {
        "crease_dataset": crease_counts,
        "waist_dataset": waist_counts,
    }
    print("Dataset preparation complete:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    setup_all_datasets()
