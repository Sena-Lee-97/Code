"""
Cell Detection using YOLOv8

- Input: Image sequence (e.g., IMG_1.png, IMG_2.png, ...)
- Output: Detection results (x, y, w, h, confidence)

Author: Sena Lee
"""

import os
import re
import time
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# Utility Functions
# =========================
def sort_image_files(file_list):
    """
    Sort files based on IMG_XXX number
    """
    return sorted(file_list, key=lambda x: int(re.search(r'IMG_(\d+)', x).group(1)))


def run_detection(model, image_path):
    """
    Run YOLO detection on a single image
    Returns: [x, y, w, h, confidence]
    """
    det_result = model(image_path, max_det=1000)

    for result in det_result:
        boxes = np.array(result.boxes.xywh.cpu())
        conf = np.array(result.boxes.conf.cpu())
        conf = np.expand_dims(conf, axis=1)

        results = np.concatenate((boxes, conf), axis=1)

    return results


def save_detection(results, save_path):
    """
    Save detection results to txt file
    """
    np.savetxt(save_path, results, fmt='%f', delimiter=',')


# =========================
# Main Pipeline
# =========================
def main(
    model_path,
    input_dir,
    output_dir
):
    """
    Run detection on all images in a directory
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Load and sort images
    files = os.listdir(input_dir)
    sorted_files = sort_image_files(files)

    start_time = time.time()

    for frame in sorted_files:
        img_path = os.path.join(input_dir, frame)

        # Detection
        results = run_detection(model, img_path)

        # Save results
        save_name = os.path.splitext(frame)[0] + ".txt"
        save_path = os.path.join(output_dir, save_name)

        save_detection(results, save_path)
        a=1
    end_time = time.time()
    print(f"Detection completed in {end_time - start_time:.2f} seconds")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    DATASET = "test_3"
    MODEL_PATH = "./weights/best.pt"
    INPUT_DIR = "./data/"+DATASET
    OUTPUT_DIR = "./results/detection_result/"+DATASET

    main(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)