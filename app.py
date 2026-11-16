from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image
import base64
import io
import gc
import torch

import oneformer
import sam3_wall

app = FastAPI(title="Wallpaper Application API")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sam3_session = {
    "wall_masks": [],
    "original_image": None
}

def decode_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)

def encode_image_base64(img_array):
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/oneformer/segment")
async def oneformer_segment(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        original_img = decode_image(image_bytes)
        original_pil = Image.fromarray(original_img)

        wall_mask, excluded_objects = oneformer.run_segmentation(original_pil)

        wall_mask_255 = (wall_mask * 255).astype(np.uint8)
        mask_b64 = encode_image_base64(wall_mask_255)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JSONResponse({
            "status": "success",
            "mask": mask_b64,
            "excluded_objects": excluded_objects,
            "pipeline": "oneformer"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ===================== FLOOR API (ADDED) =====================

@app.post("/oneformer/floor")
async def oneformer_floor(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        original_img = decode_image(image_bytes)
        original_pil = Image.fromarray(original_img)

        floor_mask = oneformer.run_floor_segmentation(original_pil)
        floor_mask_255 = (floor_mask * 255).astype(np.uint8)
        mask_b64 = encode_image_base64(floor_mask_255)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JSONResponse({
            "status": "success",
            "mask": mask_b64,
            "pipeline": "floor"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/sam3/detect")
async def sam3_detect(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        original_img = decode_image(image_bytes)

        sam3_session["original_image"] = original_img
        wall_masks, _ = sam3_wall.detect_all_walls(original_img)
        sam3_session["wall_masks"] = wall_masks

        return JSONResponse({
            "status": "success",
            "wall_count": len(wall_masks),
            "pipeline": "sam3"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/sam3/select")
async def sam3_select(
    wallpaper: UploadFile = File(...),
    click_x: int = Form(...),
    click_y: int = Form(...),
    scale: float = Form(0.25),
    opacity: int = Form(95),
    brightness: int = Form(100),
    blend_mode: str = Form("multiply")
):
    try:
        if not sam3_session["wall_masks"] or sam3_session["original_image"] is None:
            return JSONResponse({"status": "error", "message": "No walls detected. Run /sam3/detect first."}, status_code=400)

        original_img = sam3_session["original_image"]
        wall_masks = sam3_session["wall_masks"]

        orig_h, orig_w = original_img.shape[:2]
        DISPLAY_W = 600
        display_scale = DISPLAY_W / orig_w
        DISPLAY_H = int(orig_h * display_scale)

        click_x_orig = int(click_x / display_scale)
        click_y_orig = int(click_y / display_scale)

        click_x_orig = max(0, min(click_x_orig, orig_w - 1))
        click_y_orig = max(0, min(click_y_orig, orig_h - 1))

        selected_wall = None
        selected_number = None

        for wall in wall_masks:
            mask = wall["mask"]
            if mask[click_y_orig, click_x_orig] > 127:
                selected_wall = wall
                selected_number = wall["number"]
                break

        if selected_wall is None:
            min_dist = float("inf")
            for wall in wall_masks:
                mask = wall["mask"]
                moments = cv2.moments(mask)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                dist = np.sqrt((click_x_orig - cx)**2 + (click_y_orig - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    selected_wall = wall
                    selected_number = wall["number"]

        if selected_wall is None:
            return JSONResponse({"status": "error", "message": "No wall found at click location"}, status_code=400)

        sam3_session["selected_mask"] = selected_wall["mask"]
        sam3_session["selected_wall_number"] = selected_number

        selected_mask = selected_wall["mask"]
        mask_b64 = encode_image_base64(selected_mask)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JSONResponse({
            "status": "success",
            "mask": mask_b64,
            "selected_wall_number": selected_number,
            "pipeline": "sam3"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        