import os
import io
import json
import shutil
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

GITHUB_TREES = [
    {
        "name": "bhuvaneshd32/Cricket-No-Ball-Detection",
        "tree_url": "https://api.github.com/repos/bhuvaneshd32/Cricket-No-Ball-Detection/git/trees/main?recursive=1",
        "raw_base": "https://raw.githubusercontent.com/bhuvaneshd32/Cricket-No-Ball-Detection/main/",
        "legal_prefixes": [
            "Small-image-dataset/legal-ball/",
            "Final-test/Legal Ball/",
            "Large-image-dataset/training/Legal/",
            "Large-image-dataset/validation/Legal/",
            "Large-image-dataset/testing/Legal/",
        ],
        "noball_prefixes": [
            "Small-image-dataset/no-ball/",
            "Final-test/No Ball/",
            "Large-image-dataset/training/No ball/",
            "Large-image-dataset/validation/No ball/",
            "Large-image-dataset/testing/No ball/",
        ],
    },
    {
        "name": "srishikumr/NoBallDetetection",
        "tree_url": "https://api.github.com/repos/srishikumr/NoBallDetetection/git/trees/main?recursive=1",
        "raw_base": "https://raw.githubusercontent.com/srishikumr/NoBallDetetection/main/",
        "legal_prefixes": [
            "my_cricket_database/train/legal/",
            "my_cricket_database/val/legal/",
        ],
        "noball_prefixes": [
            "my_cricket_database/train/noball/",
            "my_cricket_database/val/noball/",
        ],
    },
]


def fetch_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CreaseGuard-Dataset-Fetcher/2.0",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_and_verify_image(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 512:
        return True
    try:
        encoded_path = urllib.parse.quote(url, safe=":/?&=%")
        req = urllib.request.Request(
            encoded_path, headers={"User-Agent": "CreaseGuard-Dataset-Fetcher/2.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width < 20 or img.height < 20:
            return False
        img.save(dest_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def download_real_crease_datasets(project_backend: str, max_per_class: int = 650):
    """
    Downloads real-world cricket front-foot no-ball and legal delivery images
    from open online GitHub datasets into data/raw_images/{legal,no_ball}.
    """
    raw_dir = os.path.join(project_backend, "data", "raw_images")
    legal_dir = os.path.join(raw_dir, "legal")
    noball_dir = os.path.join(raw_dir, "no_ball")
    os.makedirs(legal_dir, exist_ok=True)
    os.makedirs(noball_dir, exist_ok=True)

    legal_tasks = []
    noball_tasks = []

    for repo in GITHUB_TREES:
        print(f"Querying online dataset index: {repo['name']} ...")
        try:
            tree_data = fetch_json(repo["tree_url"])
        except Exception as e:
            print(f"  Warning: Could not query {repo['name']}: {e}")
            continue

        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            safe_name = (
                repo["name"].split("/")[0]
                + "_"
                + path.replace("/", "_").replace(" ", "_")
            )
            safe_name = os.path.splitext(safe_name)[0] + ".jpg"
            full_url = repo["raw_base"] + path

            if any(path.startswith(p) for p in repo["legal_prefixes"]):
                legal_tasks.append((full_url, os.path.join(legal_dir, safe_name)))
            elif any(path.startswith(p) for p in repo["noball_prefixes"]):
                noball_tasks.append((full_url, os.path.join(noball_dir, safe_name)))

    # Prioritize Small-image-dataset, Final-test, validation, testing, then training
    random.seed(42)
    if len(legal_tasks) > max_per_class:
        # Keep deterministic priority items + sample rest
        priority = [t for t in legal_tasks if "Large-image-dataset/training" not in t[0]]
        others = [t for t in legal_tasks if "Large-image-dataset/training" in t[0]]
        random.shuffle(others)
        legal_tasks = (priority + others)[:max_per_class]

    if len(noball_tasks) > max_per_class:
        priority = [t for t in noball_tasks if "Large-image-dataset/training" not in t[0]]
        others = [t for t in noball_tasks if "Large-image-dataset/training" in t[0]]
        random.shuffle(others)
        noball_tasks = (priority + others)[:max_per_class]

    all_tasks = [("legal", u, d) for u, d in legal_tasks] + [
        ("no_ball", u, d) for u, d in noball_tasks
    ]
    print(
        f"Downloading {len(legal_tasks)} real 'legal' and {len(noball_tasks)} real 'no_ball' cricket images..."
    )

    counts = {"legal": 0, "no_ball": 0}
    with ThreadPoolExecutor(max_workers=24) as pool:
        future_map = {
            pool.submit(download_and_verify_image, url, dest): label
            for label, url, dest in all_tasks
        }
        done_count = 0
        for fut in as_completed(future_map):
            label = future_map[fut]
            done_count += 1
            if fut.result():
                counts[label] += 1
            if done_count % 200 == 0 or done_count == len(all_tasks):
                print(
                    f"  Progress: {done_count}/{len(all_tasks)} processed "
                    f"(Legal: {counts['legal']}, No-Ball: {counts['no_ball']})"
                )

    return counts


def create_hybrid_waist_dataset(project_backend: str, num_samples_per_class: int = 600):
    """
    Creates a rich hybrid dataset for waist-high full toss (beamer) detection in
    data/raw_waist_images/{legal_waist,beamer} by combining real cricket broadcast
    backgrounds with physics-based ball trajectory & batter stance synthesis.
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

    print(
        f"Building {num_samples_per_class * 2} hybrid real-background & broadcast waist/beamer images..."
    )
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


def render_broadcast_waist_frame(is_beamer: bool, real_bg_path=None, size=(160, 160)):
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
        # Broadcast stadium & turf gradient
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

    # Popping crease line on pitch
    crease_y = random.randint(int(h * 0.76), int(h * 0.84))
    draw.line([(0, crease_y), (w, crease_y)], fill=(245, 245, 245, 210), width=3)

    # Batter stance at crease (side-on broadcast profile)
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

    # Legs / Pads
    draw.rounded_rectangle(
        [batter_x - 7, waist_y, batter_x + 9, foot_y], radius=4, fill=pad_color
    )
    # Torso
    draw.rounded_rectangle(
        [batter_x - 9, head_y + 10, batter_x + 11, waist_y + 4], radius=5, fill=jersey
    )
    # Helmet
    draw.ellipse(
        [batter_x - 7, head_y - 4, batter_x + 9, head_y + 11],
        fill=(25, 45, 95, 255),
    )
    # Bat held near waist/crease
    bat_bottom_y = waist_y + random.randint(8, 26)
    draw.line(
        [(batter_x + 6, waist_y - 2), (batter_x + 18, bat_bottom_y)],
        fill=(205, 165, 105, 255),
        width=5,
    )

    # Waist reference indicator (subtle Hawk-Eye tracking dashed plane)
    for x_seg in range(0, w, 12):
        draw.line(
            [(x_seg, waist_y), (min(x_seg + 6, w), waist_y)],
            fill=(255, 70, 70, 175),
            width=2,
        )

    # Ball trajectory & position
    ball_x = random.randint(batter_x + 18, int(w * 0.78))
    ball_radius = random.randint(4, 6)
    is_white_ball = random.random() > 0.45
    ball_color = (250, 250, 252, 255) if is_white_ball else (200, 25, 35, 255)
    trail_color = (255, 255, 255, 95) if is_white_ball else (255, 80, 80, 95)

    if is_beamer:
        # Full toss passing ABOVE waist height without bouncing
        ball_y = random.randint(max(12, head_y - 4), waist_y - 7)
        release_y = random.randint(int(h * 0.22), int(h * 0.36))
        # Draw full-toss trajectory arc from bowler end (right) to ball_x
        draw.line(
            [(w - 5, release_y), (ball_x, ball_y)],
            fill=trail_color,
            width=3,
        )
    else:
        # Legal delivery: either below waist OR bouncing off pitch
        ball_y = random.randint(waist_y + 8, crease_y - 4)
        bounce_x = min(w - 12, ball_x + random.randint(15, 45))
        bounce_y = crease_y - random.randint(0, 6)
        draw.line(
            [(w - 5, int(h * 0.28)), (bounce_x, bounce_y), (ball_x, ball_y)],
            fill=trail_color,
            width=3,
        )

    # Draw cricket ball with seam highlight
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


def augment_crease_with_synthetic_realism(project_backend: str, count_per_class: int = 200):
    """
    Adds synthetic crease frames into raw_images so the trained model excels on
    BOTH real-world broadcast images AND synthetic/simulated crease frames.
    """
    from generate_data import create_synthetic_crease_image

    raw_dir = os.path.join(project_backend, "data", "raw_images")
    legal_dir = os.path.join(raw_dir, "legal")
    noball_dir = os.path.join(raw_dir, "no_ball")
    os.makedirs(legal_dir, exist_ok=True)
    os.makedirs(noball_dir, exist_ok=True)

    for i in range(count_per_class):
        p_legal = os.path.join(legal_dir, f"synth_domain_legal_{i:04d}.jpg")
        if not os.path.exists(p_legal):
            create_synthetic_crease_image(is_no_ball=False, size=(160, 160)).save(
                p_legal, format="JPEG", quality=92
            )
        p_nb = os.path.join(noball_dir, f"synth_domain_noball_{i:04d}.jpg")
        if not os.path.exists(p_nb):
            create_synthetic_crease_image(is_no_ball=True, size=(160, 160)).save(
                p_nb, format="JPEG", quality=92
            )


def prepare_curated_samples(project_backend: str):
    """
    Creates a curated samples directory (backend/data/samples) containing real
    broadcast test images and waist frames for 1-click testing in the SaaS UI.
    """
    samples_dir = os.path.join(project_backend, "data", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    raw_legal = os.path.join(project_backend, "data", "raw_images", "legal")
    raw_noball = os.path.join(project_backend, "data", "raw_images", "no_ball")
    waist_legal = os.path.join(project_backend, "data", "raw_waist_images", "legal_waist")
    waist_beamer = os.path.join(project_backend, "data", "raw_waist_images", "beamer")

    real_legal_files = sorted(
        [
            f
            for f in os.listdir(raw_legal)
            if f.endswith(".jpg") and not f.startswith("synth_")
        ]
    )
    real_noball_files = sorted(
        [
            f
            for f in os.listdir(raw_noball)
            if f.endswith(".jpg") and not f.startswith("synth_")
        ]
    )
    waist_legal_files = sorted(
        [f for f in os.listdir(waist_legal) if f.endswith(".jpg")]
    )
    waist_beamer_files = sorted(
        [f for f in os.listdir(waist_beamer) if f.endswith(".jpg")]
    )

    if real_legal_files:
        shutil.copyfile(
            os.path.join(raw_legal, real_legal_files[0]),
            os.path.join(samples_dir, "real_legal_1.jpg"),
        )
        if len(real_legal_files) > 5:
            shutil.copyfile(
                os.path.join(raw_legal, real_legal_files[5]),
                os.path.join(samples_dir, "real_legal_2.jpg"),
            )

    if real_noball_files:
        shutil.copyfile(
            os.path.join(raw_noball, real_noball_files[0]),
            os.path.join(samples_dir, "real_noball_1.jpg"),
        )
        if len(real_noball_files) > 5:
            shutil.copyfile(
                os.path.join(raw_noball, real_noball_files[5]),
                os.path.join(samples_dir, "real_noball_2.jpg"),
            )

    if waist_legal_files:
        shutil.copyfile(
            os.path.join(waist_legal, waist_legal_files[0]),
            os.path.join(samples_dir, "waist_legal_1.jpg"),
        )
    if waist_beamer_files:
        shutil.copyfile(
            os.path.join(waist_beamer, waist_beamer_files[0]),
            os.path.join(samples_dir, "waist_beamer_1.jpg"),
        )


def setup_all_datasets(project_backend: str = None):
    if project_backend is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_backend = os.path.dirname(script_dir)

    crease_counts = download_real_crease_datasets(project_backend, max_per_class=650)
    augment_crease_with_synthetic_realism(project_backend, count_per_class=200)
    waist_counts = create_hybrid_waist_dataset(project_backend, num_samples_per_class=600)
    prepare_curated_samples(project_backend)

    summary = {
        "real_online_crease_downloaded": crease_counts,
        "total_crease_dataset": {
            "legal": len(
                os.listdir(os.path.join(project_backend, "data", "raw_images", "legal"))
            ),
            "no_ball": len(
                os.listdir(os.path.join(project_backend, "data", "raw_images", "no_ball"))
            ),
        },
        "total_waist_dataset": waist_counts,
        "sources": [r["name"] for r in GITHUB_TREES],
    }
    print("Dataset setup complete:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    setup_all_datasets()
