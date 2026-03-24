import os
import re
import cv2
import pandas as pd
from tqdm import tqdm

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TRACKER_NAMES = ["Bot_SORT", "Byte_Track", "OC_SORT"]
TRACKER_NAMES = [ "Byte_Track"]
for TRACKER_NAME in TRACKER_NAMES:
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
    def assign_track_colors(track_ids, colors):
        if len(colors) == 0:
            raise ValueError("COLOR_LIST must contain at least one color.")
        unique_ids = sorted(track_ids)
        return {tid: colors[i % len(colors)] for i, tid in enumerate(unique_ids)}


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


    def load_tracking_txt(txt_path):
        """
        Expected format per line:
        track_id,frame,x,y,w,h
        """
        df = pd.read_csv(
            txt_path,
            header=None,
            names=["track_id", "frame", "x", "y", "w", "h"]
        )
        df["track_id"] = df["track_id"].astype(int)
        df["frame"] = df["frame"].astype(int)
        return df


    def draw_tracking_overlay(image, frame_tracks, track_colors):
        output = image.copy()

        for _, row in frame_tracks.iterrows():
            track_id = int(row["track_id"])
            x = int(round(row["x"]))
            y = int(round(row["y"]))
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
            print(f"\n===== Processing {TRACKER_NAME} / {DATASET} =====")

            IMAGE_DIR = os.path.join(BASE_DIR, "..", "data", DATASET)
            TRACK_TXT_PATH = os.path.join(
                BASE_DIR, "..", "results", "baseline_tracking",
                TRACKER_NAME, DATASET, f"{DATASET}.txt"
            )
            SAVE_DIR = os.path.join(
                BASE_DIR, "..", "results", "baseline_tracking",
                TRACKER_NAME, DATASET
            )
            VIDEO_SAVE_PATH = os.path.join(
                SAVE_DIR, f"{DATASET}_tracking_overlay_step10.mp4"
            )

            if not os.path.isdir(IMAGE_DIR):
                print(f"Skip: image folder not found -> {IMAGE_DIR}")
                continue

            if not os.path.isfile(TRACK_TXT_PATH):
                print(f"Skip: tracking txt not found -> {TRACK_TXT_PATH}")
                continue

            os.makedirs(SAVE_DIR, exist_ok=True)

            # Load tracking txt
            df_track = load_tracking_txt(TRACK_TXT_PATH)
            track_colors = assign_track_colors(df_track["track_id"].unique(), COLOR_LIST)

            # Collect actual image files
            image_files_raw = [
                f for f in os.listdir(IMAGE_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            image_files = sort_image_files(image_files_raw)

            if len(image_files) == 0:
                print(f"Skip: no image files found -> {IMAGE_DIR}")
                continue

            # Frame numbers available in tracking txt
            valid_frame_numbers = set(df_track["frame"].unique())

            # Match image files with tracked frames
            valid_pairs = []
            for frame_num, filename in image_files:
                if frame_num in valid_frame_numbers:
                    valid_pairs.append((frame_num, filename))

            if len(valid_pairs) == 0:
                print(f"Skip: no matching image files for tracked frame indices -> {DATASET}")
                continue

            # Apply frame skip
            selected_pairs = valid_pairs[::STEP]

            if len(selected_pairs) == 0:
                print(f"Skip: no frames selected after STEP -> {DATASET}")
                continue

            # Read first valid image
            _, first_filename = selected_pairs[0]
            first_image_path = os.path.join(IMAGE_DIR, first_filename)
            first_image = cv2.imread(first_image_path)

            if first_image is None:
                print(f"Skip: failed to read first image -> {first_image_path}")
                continue

            height, width = first_image.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, OUTPUT_FPS, (width, height))

            if not writer.isOpened():
                print(f"Skip: failed to open VideoWriter -> {VIDEO_SAVE_PATH}")
                continue

            # Save video
            for frame_num, filename in tqdm(selected_pairs, desc=f"{TRACKER_NAME}-{DATASET}", unit="frame"):
                image_path = os.path.join(IMAGE_DIR, filename)
                image = cv2.imread(image_path)

                if image is None:
                    tqdm.write(f"Warning: failed to load {image_path}")
                    continue

                frame_tracks = df_track[df_track["frame"] == frame_num]
                overlay = draw_tracking_overlay(image, frame_tracks, track_colors)
                writer.write(overlay)

            writer.release()
            print(f"Saved video: {VIDEO_SAVE_PATH}")


    if __name__ == "__main__":
        main()