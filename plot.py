import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from ultralytics.utils.metrics import plot_pr_curve

# ==============================
# CONFIG
# ==============================
folder_path = r"D:\Downloads\pr_curve_values\pr_curve_values"
save_path = Path("PR_curve_ultralytics2.png")
# ==============================

csv_files = sorted(glob.glob(os.path.join(folder_path, "PR_curve_class_*.csv")))

# 1000 recall points exactly like Ultralytics
px = np.linspace(0, 1, 1000)

py_list = []
ap_list = []
names = {}

for idx, file in enumerate(csv_files):

    class_name = os.path.basename(file).replace("PR_curve_class_", "").replace(".csv", "")
    names[idx] = class_name

    df = pd.read_csv(file).sort_values("Recall")

    recall = df["Recall"].values
    precision = df["Precision"].values

    # Remove duplicate recall values
    recall_unique = np.unique(recall)
    precision_unique = np.array([precision[recall == r].max() for r in recall_unique])

    # Precision envelope (VERY IMPORTANT — matches compute_ap)
    precision_unique = np.flip(np.maximum.accumulate(np.flip(precision_unique)))

    # Add sentinel values like compute_ap()
    mrec = np.concatenate(([0.0], recall_unique, [1.0]))
    mpre = np.concatenate(([1.0], precision_unique, [0.0]))

    # Interpolate to 1000 points
    precision_interp = np.interp(px, mrec, mpre)

    # Compute AP exactly like compute_ap (101-point COCO)
    x_101 = np.linspace(0, 1, 101)
    ap50 = np.trapezoid(np.interp(x_101, mrec, mpre), x_101)

    # Ultralytics expects ap shape (nc, 10)
    ap_per_class = np.zeros(10)
    ap_per_class[0] = ap50  # only AP@0.5
    ap_list.append(ap_per_class)

    py_list.append(precision_interp)

# Convert to numpy
py = np.array(py_list)        # shape (nc, 1000)
ap = np.array(ap_list)        # shape (nc, 10)

# ==============================
# Plot Using Official Ultralytics Function
# ==============================
plot_pr_curve(
    px=px,
    py=py,
    ap=ap,
    save_dir=save_path,
    names=names
)

print("Saved:", save_path)