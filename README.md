# Wallpaper Preview Application - README.md (Final Version)

**Project Name:** Wallpaper Preview App  
**Version:** Final  
**Date:** January 2026  

A modern web-based tool that lets users upload a photo of their room and instantly visualize how a chosen wallpaper pattern would look on the walls using advanced AI segmentation.

## Key Features

- **Two AI-Powered Modes**:
  - **Automatic Mode** (OneFormer): Instantly detects all walls and applies the wallpaper automatically.
  - **Click-to-Select Mode** (SAM3): Detects multiple walls and lets you click on a specific wall to apply wallpaper only there.

- **Real-time Adjustments**:
  - Pattern scale/size (1–100)
  - Overall room brightness/lighting (-100 to +100)
  - Live preview updates as you adjust sliders

- **Export**: Download the final preview as a PNG image.

- **Clean, responsive UI** with mobile support.

## Prerequisites

- Python 3.8 or higher
- GPU with CUDA recommended for speed (CPU works but slower)
- At least 8 GB RAM (16 GB+ recommended)

## Folder Structure (Final)

```
wallpaper-app/
├── app.py                # FastAPI backend server
├── index.html            # Frontend web interface
├── oneformer.py          # OneFormer segmentation logic
├── sam3_wall.py          # SAM3 wall detection logic
├── sam3.pt               # SAM3 model checkpoint (must download – see below)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Create Project Folder
```bash
mkdir wallpaper-app
cd wallpaper-app
```

### 2. Save All Provided Files
Place the following files exactly as named:
- `app.py`
- `index.html`
- `oneformer.py`
- `sam3_wall.py`
- `requirements.txt`
- This `README.md`

### 3. Download the SAM3 Model Checkpoint (sam3.pt)

The SAM3 (Segment Anything Model 3) checkpoint file (`sam3.pt`) is hosted on Hugging Face and **requires access approval from Meta**.

**Steps to Download:**

1. Go to the official model page: https://huggingface.co/facebook/sam3
2. Click **"Request Access"** (you'll need to agree to the terms and provide contact info).
3. Wait for approval (usually within 24 hours or less).
4. Once approved and logged in to Hugging Face, download the file `sam3.pt` directly from the "Files" tab  
   (or use this direct link after approval: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt).

For more details, check the official GitHub repo: https://github.com/facebookresearch/sam3

**Important:** Place the downloaded `sam3.pt` in the project root folder (same level as `app.py`) to run the wallpaper app's SAM3 mode.

> The OneFormer model will be automatically downloaded from Hugging Face on first use — no manual download needed.

### 4. Create and Activate Virtual Environment (Strongly Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` in your terminal prompt.

### 5. Install Dependencies
With the virtual environment activated:
```bash
pip install -r requirements.txt
```

> For faster performance with a CUDA-capable NVIDIA GPU, you can replace the torch lines in `requirements.txt` with CUDA versions and reinstall.

## How to Run the Application

### Step 1: Start the Backend Server
In your terminal (with virtual environment activated):
```bash
python app.py
```

Server runs on http://localhost:8000

### Step 2: Open the Frontend

**Recommended (Local Server):**
Open a second terminal, activate the virtual environment, then:
```bash
python -m http.server 8080
```
Visit: http://localhost:8080/index.html

**Alternative:** Simply open `index.html` directly in your browser.

## How to Use the App

### Automatic Mode (Recommended for Most Users)

1. Keep **"Automatic"** selected (default).
2. Upload a room photo (Step 1).
3. Upload a wallpaper pattern (Step 2).
4. Wallpaper is automatically applied to detected walls.
5. Adjust **Pattern Size** and **Room Lighting** sliders.
6. Click **Download** to save the preview.

### Click-to-Select Mode (For Precision)

1. Select **"Click to Select"** mode.
2. Upload room photo and wallpaper pattern.
3. Click on the specific wall you want to decorate (cursor becomes crosshair).
4. Adjust sliders and download.

## Tips for Best Results

- Use clear, well-lit, straight-on room photos.
- Avoid extreme angles or heavy clutter.
- Resize very large images (>2000px) before uploading for faster processing.

## Troubleshooting

| Issue                        | Solution                                                                 |
|------------------------------|--------------------------------------------------------------------------|
| SAM3 mode not working        | Ensure `sam3.pt` is in the root folder and correctly downloaded           |
| Server errors / model load   | Check terminal output; ensure virtual env is active and deps installed   |
| Slow performance             | Use GPU if available; try smaller images                                 |
| No walls detected            | Try a clearer photo or switch to Automatic mode                          |

## Deactivating Virtual Environment

```bash
deactivate
```

## Credits & Licenses

- Built using:
  - **OneFormer** by SHI Labs
  - **SAM3** (Segment Anything Model 3) by Meta AI
  - FastAPI, PyTorch, OpenCV

Please respect the original model licenses.

