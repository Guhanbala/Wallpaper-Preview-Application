import cv2
import torch
import numpy as np
import gc
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def remove_shadows(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result

def preprocess_for_segformer(img):
    img = remove_shadows(img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(img, (0, 0), 0.5)
    img = cv2.addWeighted(img, 1.1, blur, -0.1, 0)
    return img

SAM3_CHECKPOINT = "sam3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

DISPLAY_W, DISPLAY_H = 900, 600

WALL_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 200, 100), (200, 100, 255)
]

print("Loading SAM 3...")
model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
model.to(DEVICE)
model.eval()
processor = Sam3Processor(model)
print("SAM 3 loaded!")

def snap_to_axis(contour, tolerance=18):
    if len(contour) < 2:
        return contour

    pts = contour[:, 0, :].astype(float)

    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]

        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])

        if dx < tolerance:
            avg_x = (p1[0] + p2[0]) / 2
            pts[i][0] = avg_x
            pts[(i + 1) % len(pts)][0] = avg_x

        if dy < tolerance:
            avg_y = (p1[1] + p2[1]) / 2
            pts[i][1] = avg_y
            pts[(i + 1) % len(pts)][1] = avg_y

    return pts.astype(np.int32).reshape(-1, 1, 2)

def detect_all_walls(image_np):
    wall_masks = []

    if image_np is None:
        return [], None

    print("Detecting walls...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orig_h, orig_w = image_np.shape[:2]

    MAX_DIM = 1024
    scale = 1.0
    model_input_np = image_np

    if max(orig_h, orig_w) > MAX_DIM:
        scale = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        model_input_np = cv2.resize(image_np, (new_w, new_h))
        print(f"Resized for processing: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    pil_image = Image.fromarray(model_input_np)

    try:
        with torch.inference_mode():
            state = processor.set_image(pil_image)
            results_wall = processor.set_text_prompt(state=state, prompt="interior painted wall")
            masks_wall = results_wall.get("masks", None)
            scores_wall = results_wall.get("scores", [])

        if masks_wall is None or masks_wall.numel() == 0:
            print("No wall masks detected.")
            return [], None

        print(f"Processing {masks_wall.shape[0]} candidates...")

        for i in range(masks_wall.shape[0]):
            low_res_mask = masks_wall[i].squeeze().cpu().numpy()
            score = float(scores_wall[i]) if i < len(scores_wall) else 0.0

            if score < 0.35:
                continue

            mask_uint8 = (low_res_mask > 0.28).astype(np.uint8) * 255
            full_mask = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            kernel = np.ones((9, 9), np.uint8)
            full_mask = cv2.dilate(full_mask, kernel, iterations=1)
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            if cv2.countNonZero(full_mask) < 15000:
                continue

            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_canvas = np.zeros_like(full_mask)

            for cnt in contours:
                if cv2.contourArea(cnt) < 15000:
                    continue

                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                rectified = snap_to_axis(approx, tolerance=18)
                cv2.drawContours(clean_canvas, [rectified], -1, 255, -1)

            clean_canvas = cv2.morphologyEx(clean_canvas, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

            for existing in wall_masks:
                clean_canvas[existing["mask"] > 0] = 0

            if cv2.countNonZero(clean_canvas) > 15000:
                wall_masks.append({
                    "mask": clean_canvas,
                    "color": WALL_COLORS[i % len(WALL_COLORS)],
                    "number": len(wall_masks) + 1,
                    "score": score
                })

        overlay_img = draw_overlays(image_np, wall_masks)
        return wall_masks, overlay_img

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU Out of Memory. The image was too large for VRAM.")
    except Exception as e:
        raise RuntimeError(str(e))

def draw_overlays(image_np, wall_masks):
    if image_np is None or not wall_masks:
        return None

    display_img = cv2.resize(image_np, (DISPLAY_W, DISPLAY_H))
    overlay_pil = Image.fromarray(display_img).convert("RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    for wall in wall_masks:
        mask_resized = cv2.resize(wall["mask"], (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 127).astype(np.uint8)

        color_rgba = wall["color"] + (85,)
        colored_layer = np.zeros((DISPLAY_H, DISPLAY_W, 4), dtype=np.uint8)
        colored_layer[mask_resized == 1] = color_rgba

        overlay_pil = Image.alpha_composite(overlay_pil, Image.fromarray(colored_layer, "RGBA"))

        coords = np.where(mask_resized == 1)
        if len(coords[0]) > 0:
            cy = int(np.mean(coords[0]))
            cx = int(np.mean(coords[1]))
            draw.text((cx - 12, cy - 20), str(wall['number']), fill="white")

    return np.array(overlay_pil.convert("RGB"))

def select_wall_by_click(image_np, wall_masks, click_x, click_y):
    if not wall_masks or image_np is None:
        return None

    orig_h, orig_w = image_np.shape[:2]
    click_x = int(click_x * orig_w / DISPLAY_W)
    click_y = int(click_y * orig_h / DISPLAY_H)

    selected = None
    min_dist = float("inf")

    for wall in wall_masks:
        mask = wall["mask"]
        if (
            click_y < mask.shape[0] and
            click_x < mask.shape[1] and
            mask[click_y, click_x] > 127
        ):
            return wall

    for wall in wall_masks:
        mask = wall["mask"]
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        dist = np.hypot(click_x - cx, click_y - cy)

        if dist < min_dist:
            min_dist = dist
            selected = wall

    return selected

def highlight_wall(image_np, wall):
    mask = (wall["mask"] // 255).astype(np.uint8)
    highlight = image_np.copy()
    highlight[mask == 1] = [70, 255, 70]

    blended = cv2.addWeighted(image_np, 0.6, highlight, 0.4, 0)
    resized = cv2.resize(blended, (DISPLAY_W, DISPLAY_H))
    return resized

def load_image(path):
    img = Image.open(path).convert("RGB")
    image_np = preprocess_for_segformer(np.array(img))
    return image_np
