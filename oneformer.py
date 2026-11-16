import cv2
import torch
import numpy as np
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_large"
MAX_INFERENCE_DIM = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

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

print(f"Loading {ONEFORMER_MODEL} on {DEVICE}...")
processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
model = OneFormerForUniversalSegmentation.from_pretrained(
    ONEFORMER_MODEL, use_safetensors= False
)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

def run_segmentation(image_pil):
    try:
        image_np = np.array(image_pil)
        image_np = preprocess_for_segformer(image_np)
        input_pil = Image.fromarray(image_np)

        w, h = input_pil.size
        scale = 1.0
        if max(w, h) > MAX_INFERENCE_DIM:
            scale = MAX_INFERENCE_DIM / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            input_pil = input_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

        inputs = processor(
            images=input_pil,
            task_inputs=["semantic"],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[input_pil.size[::-1]]
        )[0]

        seg_map_np = seg_map.cpu().numpy()
        id2label = model.config.id2label

        detected_ids = np.unique(seg_map_np)
        excluded_objects = []
        for class_id in detected_ids:
            if class_id == 0:
                continue
            label_name = id2label.get(class_id, f"Unknown_ID_{class_id}")
            excluded_objects.append(f"{label_name} (ID: {class_id})")

        detected_objects_text = ", ".join(excluded_objects)

        wall_binary = (seg_map_np == 0).astype(np.uint8)

        if scale != 1.0:
            wall_binary = cv2.resize(wall_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((5,5), np.uint8)
        wall_binary = cv2.morphologyEx(wall_binary, cv2.MORPH_OPEN, kernel)
        wall_binary = cv2.morphologyEx(wall_binary, cv2.MORPH_CLOSE, kernel)

        return wall_binary, detected_objects_text

    except Exception as e:
        raise RuntimeError(f"Segmentation Error: {e}")

# ===================== FLOOR SEGMENTATION (ADDED) =====================

def run_floor_segmentation(image_pil):
    try:
        image_np = np.array(image_pil)
        image_np = preprocess_for_segformer(image_np)
        input_pil = Image.fromarray(image_np)

        w, h = input_pil.size
        scale = 1.0
        if max(w, h) > MAX_INFERENCE_DIM:
            scale = MAX_INFERENCE_DIM / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            input_pil = input_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

        inputs = processor(
            images=input_pil,
            task_inputs=["semantic"],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[input_pil.size[::-1]]
        )[0]

        seg_map_np = seg_map.cpu().numpy()
        id2label = model.config.id2label

        floor_ids = [k for k,v in id2label.items() if "floor" in v.lower()]
        floor_binary = np.isin(seg_map_np, floor_ids).astype(np.uint8)

        if scale != 1.0:
            floor_binary = cv2.resize(floor_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((5,5), np.uint8)
        floor_binary = cv2.morphologyEx(floor_binary, cv2.MORPH_OPEN, kernel)
        floor_binary = cv2.morphologyEx(floor_binary, cv2.MORPH_CLOSE, kernel)

        return floor_binary

    except Exception as e:
        raise RuntimeError(f"Floor Segmentation Error: {e}")
