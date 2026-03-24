import os
import re
import cv2
import pandas as pd
from tqdm import tqdm

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["test_1", "test_2", "test_3"]

STEP = 10
OUTPUT_FPS = 60
DRAW_TEXT = True
DRAW_MARKER = True
MARKER_RADIUS = 3
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1

COLOR_LIST = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 215, 0),
    (255, 50, 147),
    (135, 206, 250),
    (255, 218, 185),
]

# =========================
# Utility functions
# =========================
def assign_track_colors(num_ids, colors):
    return {i + 1: colors[i % len(colors)] for i in range(num_ids)}

def extract_frame_number(filename):
    match = re.search(r"IMG_(\d+)", filename)
    return int(match.group(1)) if match else None

def sort_image_files(file_list):
    valid_files = []
    for f in file_list:
        frame_num = extract_frame_number(f)
        if frame_num is not None:
            valid_files.append((frame_num, f))
    valid_files.sort(key=lambda x: x[0])
    return valid_files

def draw_tracking_overlay(image, df_track, frame_idx, track_colors):
    output = image.copy()

    for col in range(0, len(df_track.columns), 2):
        x = df_track.iloc[frame_idx, col]
        y = df_track.iloc[frame_idx, col + 1]
        track_id = (col // 2) + 1

        if pd.isna(x) or pd.isna(y):
            continue

        x = int(round(x))
        y = int(round(y))
        color = track_colors[track_id]

        if DRAW_MARKER:
            cv2.circle(output, (x, y), MARKER_RADIUS, color, -1)

        if DRAW_TEXT:
            cv2.putText(
                output,
                str(track_id),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                color,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    return output


# =========================
# Main
# =========================
def main():

    for DATASET in DATASETS:
        print(f"\n===== Processing {DATASET} =====")

        IMAGE_DIR = os.path.join(BASE_DIR, "..", "data", DATASET)
        RESULT_DIR = os.path.join(BASE_DIR, "..", "results", "tracking_result", DATASET)
        TRACKING_CSV_PATH = os.path.join(RESULT_DIR, "save_files", f"{DATASET}cell_xy.csv")
        VIDEO_SAVE_PATH = os.path.join(RESULT_DIR, "save_track", "tracking_overlay_step10.mp4")

        if not os.path.isdir(IMAGE_DIR):
            print(f"Skip: Image folder not found -> {IMAGE_DIR}")
            continue

        if not os.path.isfile(TRACKING_CSV_PATH):
            print(f"Skip: CSV not found -> {TRACKING_CSV_PATH}")
            continue

        os.makedirs(os.path.dirname(VIDEO_SAVE_PATH), exist_ok=True)

        df_track = pd.read_csv(TRACKING_CSV_PATH, header=None)
        num_ids = len(df_track.columns) // 2
        track_colors = assign_track_colors(num_ids, COLOR_LIST)

        image_files_raw = [
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files = sort_image_files(image_files_raw)

        valid_pairs = []
        num_csv_frames = len(df_track)

        for frame_num, filename in image_files:
            if frame_num < num_csv_frames:
                valid_pairs.append((frame_num, filename))

        selected_pairs = valid_pairs[::STEP]

        first_frame_num, first_filename = selected_pairs[0]
        first_image = cv2.imread(os.path.join(IMAGE_DIR, first_filename))
        height, width = first_image.shape[:2]

        writer = cv2.VideoWriter(
            VIDEO_SAVE_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            OUTPUT_FPS,
            (width, height)
        )

        for frame_num, filename in tqdm(selected_pairs, desc=DATASET):
            image = cv2.imread(os.path.join(IMAGE_DIR, filename))
            if image is None:
                continue

            overlay = draw_tracking_overlay(image, df_track, frame_num, track_colors)
            writer.write(overlay)

        writer.release()
        print(f"Saved: {VIDEO_SAVE_PATH}")


if __name__ == "__main__":
    main()