import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_crease_image(is_no_ball, size=(128, 128)):
    # 1. Create background (simulating grass/pitch)
    bg_r = random.randint(30, 80)
    bg_g = random.randint(100, 180)
    bg_b = random.randint(30, 80)
    img = Image.new("RGB", size, (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    np_img = np.array(img)
    noise = np.random.normal(0, 8, np_img.shape).astype(np.int16)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(img)
    
    crease_y = random.randint(75, 85)
    line_thickness = random.randint(4, 7)
    line_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
    draw.line([(0, crease_y), (size[0], crease_y)], fill=line_color, width=line_thickness)
    
    shoe_width = random.randint(18, 26)
    shoe_length = random.randint(35, 48)
    
    if is_no_ball:
        y_top = crease_y + random.randint(1, 15)
    else:
        y_top = crease_y - random.randint(10, 45)
        
    y_bottom = y_top + shoe_length
    x_center = size[0] // 2 + random.randint(-20, 20)
    x_left = x_center - shoe_width // 2
    x_right = x_center + shoe_width // 2
    
    shoe_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    
    shoe_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    shoe_draw = ImageDraw.Draw(shoe_layer)
    shoe_draw.ellipse([x_left, y_top, x_right, y_bottom], fill=shoe_color)
    
    accent_color = (random.randint(0, 100), random.randint(0, 100), random.randint(150, 255))
    shoe_draw.rectangle([x_center - 2, y_top + 15, x_center + 2, y_top + 25], fill=accent_color)
    
    angle = random.randint(-15, 15)
    rotated_shoe = shoe_layer.rotate(angle, resample=Image.BICUBIC, center=(x_center, (y_top + y_bottom) // 2))
    
    img.paste(rotated_shoe, (0, 0), rotated_shoe)
    
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        
    return img

def create_synthetic_waist_image(is_beamer, size=(128, 128)):
    # Simulating batter at crease side-on
    bg_r, bg_g, bg_b = random.randint(100, 150), random.randint(100, 150), random.randint(150, 200) # Sky/stadium background
    img = Image.new("RGB", size, (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    # Draw pitch (bottom)
    draw.rectangle([(0, 100), (size[0], size[1])], fill=(80, 120, 80))
    
    # Draw batter body (simple rectangle/lines)
    batter_x = 30
    batter_waist_y = 60
    # Torso
    draw.rectangle([(batter_x, 30), (batter_x+15, batter_waist_y+30)], fill=(200, 200, 200))
    # Waist line indicator (just for visual representation, in reality model learns from body proportions)
    draw.line([(batter_x-5, batter_waist_y), (batter_x+20, batter_waist_y)], fill=(255, 0, 0), width=2)
    
    # Draw the ball (small white circle)
    ball_x = random.randint(50, 100) # somewhere in front of batter
    if is_beamer:
        # Ball is above waist (smaller y value)
        ball_y = random.randint(10, batter_waist_y - 5)
    else:
        # Ball is below waist or bouncing
        ball_y = random.randint(batter_waist_y + 10, 95)
        
    draw.ellipse([(ball_x, ball_y), (ball_x+6, ball_y+6)], fill=(255, 255, 255))
    
    # Blur
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        
    return img

def generate_dataset(num_samples_per_class=500, base_dir="data/mock", waist_dir="data/waist_mock"):
    os.makedirs(os.path.join(base_dir, "legal"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "no_ball"), exist_ok=True)
    
    os.makedirs(os.path.join(waist_dir, "legal_waist"), exist_ok=True)
    os.makedirs(os.path.join(waist_dir, "beamer"), exist_ok=True)
    
    print(f"Generating {num_samples_per_class} synthetic crease images for 'legal' class...")
    for i in range(num_samples_per_class):
        img = create_synthetic_crease_image(is_no_ball=False)
        img.save(os.path.join(base_dir, "legal", f"legal_{i:04d}.png"))
        
    print(f"Generating {num_samples_per_class} synthetic crease images for 'no_ball' class...")
    for i in range(num_samples_per_class):
        img = create_synthetic_crease_image(is_no_ball=True)
        img.save(os.path.join(base_dir, "no_ball", f"no_ball_{i:04d}.png"))
        
    print(f"Generating {num_samples_per_class} synthetic waist images for 'legal_waist' class...")
    for i in range(num_samples_per_class):
        img = create_synthetic_waist_image(is_beamer=False)
        img.save(os.path.join(waist_dir, "legal_waist", f"legal_waist_{i:04d}.png"))
        
    print(f"Generating {num_samples_per_class} synthetic waist images for 'beamer' class...")
    for i in range(num_samples_per_class):
        img = create_synthetic_waist_image(is_beamer=True)
        img.save(os.path.join(waist_dir, "beamer", f"beamer_{i:04d}.png"))
        
    print("Synthetic dataset generation complete for both Crease and Waist models!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_backend = os.path.dirname(script_dir)
    mock_dir = os.path.join(project_backend, "data", "mock")
    waist_dir = os.path.join(project_backend, "data", "waist_mock")
    generate_dataset(num_samples_per_class=500, base_dir=mock_dir, waist_dir=waist_dir)
