"""
Run baseline trackers (Bot-SORT, ByteTrack, OC-SORT) with YOLO detections
and save tracking outputs + performance summaries.

Author: Sena Lee
"""

import os
import re
import time
import json
from pathlib import Path

import numpy as np
import psutil
from ultralytics import YOLO
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# ==============================
# Configuration
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "weights", "best.pt")
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "results", "baseline_tracking")

DATASETS = ["test_1", "test_2", "test_3"]

TRACKER_CONFIGS = {
    "Bot_SORT": "botsort.yaml",
    "Byte_Track": "bytetrack.yaml",
    "OC_SORT": "ocsort.yaml",
}

MAX_DET = 1000
SAVE_FRAME_TIME = True
SAVE_DETAILED_NUMPY = False
USE_GPU_INDEX = 0


# ==============================
# Utility functions
# ==============================

def extract_frame_number(filename: str):
    match = re.search(r"IMG_(\d+)", filename)
    return int(match.group(1)) if match else None


def sort_image_files(file_list):
    """Sort image files by IMG_<number>."""
    def extract_num(filename):
        match = re.search(r"IMG_(\d+)", filename)
        return int(match.group(1)) if match else float("inf")

    return sorted(file_list, key=extract_num)


def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device(f"cuda:{USE_GPU_INDEX}")
    return torch.device("cpu")


def init_nvml():
    """Initialize NVML if available."""
    if not NVML_AVAILABLE:
        return None

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(USE_GPU_INDEX)
        return handle
    except Exception:
        return None


def shutdown_nvml():
    if NVML_AVAILABLE:
        try:
            nvmlShutdown()
        except Exception:
            pass


def sample_system_usage(process, gpu_handle=None):
    """Collect CPU/GPU utilization and memory usage."""
    cpu_util = psutil.cpu_percent(interval=None)

    rss_mb = process.memory_info().rss / (1024 ** 2)
    try:
        uss_mb = process.memory_full_info().uss / (1024 ** 2)
    except Exception:
        uss_mb = None

    gpu_util = None
    gpu_mem_mb = None

    if gpu_handle is not None:
        try:
            util = nvmlDeviceGetUtilizationRates(gpu_handle)
            mem = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_util = float(util.gpu)
            gpu_mem_mb = float(mem.used / (1024 ** 2))
        except Exception:
            pass

    return {
        "cpu_util": cpu_util,
        "rss_mb": rss_mb,
        "uss_mb": uss_mb,
        "gpu_util": gpu_util,
        "gpu_mem_mb": gpu_mem_mb,
    }


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_latency_stats(frame_times):
    if len(frame_times) == 0:
        return {
            "mean_sec": None,
            "std_sec": None,
            "p50_sec": None,
            "p90_sec": None,
            "p95_sec": None,
        }

    arr = np.array(frame_times, dtype=float)
    return {
        "mean_sec": float(np.mean(arr)),
        "std_sec": float(np.std(arr)),
        "p50_sec": float(np.percentile(arr, 50)),
        "p90_sec": float(np.percentile(arr, 90)),
        "p95_sec": float(np.percentile(arr, 95)),
    }


def summarize_usage(samples):
    """Aggregate CPU/GPU usage samples."""
    if len(samples) == 0:
        return {
            "cpu": {},
            "gpu": {},
        }

    cpu_utils = [s["cpu_util"] for s in samples if s["cpu_util"] is not None]
    rss_vals = [s["rss_mb"] for s in samples if s["rss_mb"] is not None]
    uss_vals = [s["uss_mb"] for s in samples if s["uss_mb"] is not None]
    gpu_utils = [s["gpu_util"] for s in samples if s["gpu_util"] is not None]
    gpu_mems = [s["gpu_mem_mb"] for s in samples if s["gpu_mem_mb"] is not None]

    return {
        "cpu": {
            "avg_util_percent": float(np.mean(cpu_utils)) if cpu_utils else None,
            "avg_rss_mb": float(np.mean(rss_vals)) if rss_vals else None,
            "peak_rss_mb": float(np.max(rss_vals)) if rss_vals else None,
            "peak_uss_mb": float(np.max(uss_vals)) if uss_vals else None,
        },
        "gpu": {
            "avg_util_percent": float(np.mean(gpu_utils)) if gpu_utils else None,
            "max_util_percent": float(np.max(gpu_utils)) if gpu_utils else None,
            "avg_mem_mb": float(np.mean(gpu_mems)) if gpu_mems else None,
            "peak_mem_mb": float(np.max(gpu_mems)) if gpu_mems else None,
        },
    }


def get_torch_gpu_peaks():
    """Get PyTorch peak GPU memory if available."""
    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        return {
            "torch_alloc_peak_mb": None,
            "torch_reserved_peak_mb": None,
        }

    try:
        return {
            "torch_alloc_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "torch_reserved_peak_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
        }
    except Exception:
        return {
            "torch_alloc_peak_mb": None,
            "torch_reserved_peak_mb": None,
        }


def write_tracking_txt(track_id_boxes, save_path):
    """Save tracked coordinates in txt format."""
    with open(save_path, "w", encoding="utf-8") as f:
        for track_id in sorted(track_id_boxes.keys()):
            for box in track_id_boxes[track_id]:
                f.write(",".join(map(str, box)) + "\n")


def extract_track_results(result):
    """Extract boxes and track IDs safely."""
    if result is None or len(result) == 0:
        return [], []

    boxes_obj = result[0].boxes
    if boxes_obj is None or boxes_obj.xywh is None:
        return [], []

    boxes = boxes_obj.xywh.cpu().numpy()

    if boxes_obj.id is None:
        return boxes, []

    track_ids = boxes_obj.id.int().cpu().tolist()
    return boxes, track_ids


def filter_full_length_tracks(track_id_boxes, total_frames):
    """
    Keep only tracks that:
    - start at frame 0
    - end at frame total_frames - 1
    - appear in every frame exactly once
    """
    filtered_track_id_boxes = {}
    expected_frames = list(range(total_frames))

    for track_id, boxes in track_id_boxes.items():
        if len(boxes) == 0:
            continue

        frames = [int(b[1]) for b in boxes]
        frames_sorted = sorted(frames)

        if min(frames_sorted) != 0:
            continue
        if max(frames_sorted) != total_frames - 1:
            continue
        if len(frames_sorted) != total_frames:
            continue
        if frames_sorted != expected_frames:
            continue

        filtered_track_id_boxes[track_id] = boxes

    return filtered_track_id_boxes


# ==============================
# Main tracker runner
# ==============================

def run_single_tracker_dataset(model, tracker_name, tracker_yaml, dataset_name, gpu_handle):
    image_dir = Path(DATA_ROOT) / dataset_name
    output_dir = Path(OUTPUT_ROOT) / tracker_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_txt_path = output_dir / f"{dataset_name}.txt"
    summary_json_path = output_dir / "performance_summary.json"

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files = sort_image_files(image_files)

    if len(image_files) == 0:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    process = psutil.Process(os.getpid())

    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.set_device(USE_GPU_INDEX)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    track_id_boxes = {}
    frame_times = []
    usage_samples = []

    start_time = time.time()

    for frame_file in tqdm(image_files, desc=f"{tracker_name} - {dataset_name}", unit="frame"):
        frame_path = str(image_dir / frame_file)

        frame_idx = extract_frame_number(frame_file)
        if frame_idx is None:
            continue

        frame_start = time.time()
        usage_samples.append(sample_system_usage(process, gpu_handle))

        result = model.track(
            frame_path,
            persist=True,
            max_det=MAX_DET,
            tracker=tracker_yaml,
            verbose=False,
            device=USE_GPU_INDEX,
        )

        boxes, track_ids = extract_track_results(result)

        if len(track_ids) > 0 and len(boxes) > 0:
            for box, track_id in zip(boxes, track_ids):
                if track_id not in track_id_boxes:
                    track_id_boxes[track_id] = []

                track_id_boxes[track_id].append(
                    [int(track_id), int(frame_idx), *box.tolist()]
                )

        frame_times.append(time.time() - frame_start)

    # ==============================
    # Track filtering (moved OUTSIDE frame loop)
    # ==============================
    total_frames = len(image_files)
    track_id_boxes = filter_full_length_tracks(track_id_boxes, total_frames)

    total_time = time.time() - start_time
    throughput_fps = len(image_files) / total_time if total_time > 0 else None

    latency_stats = compute_latency_stats(frame_times)
    usage_summary = summarize_usage(usage_samples)
    torch_peaks = get_torch_gpu_peaks()

    summary = {
        "tracker": tracker_name,
        "dataset": dataset_name,
        "model_path": MODEL_PATH,
        "tracker_yaml": tracker_yaml,
        "num_frames": len(image_files),
        "execution_time_sec": total_time,
        "throughput_fps": throughput_fps,
        "num_tracked_ids_after_filtering": len(track_id_boxes),
        "filtering_rule": {
            "start_frame_must_be": 0,
            "end_frame_must_be": total_frames - 1,
            "must_cover_all_frames": True,
            "must_have_no_missing_frames": True,
        },
        "latency": latency_stats,
        "cpu": usage_summary["cpu"],
        "gpu": {
            **usage_summary["gpu"],
            **torch_peaks,
        },
    }

    write_tracking_txt(track_id_boxes, tracking_txt_path)
    save_json(summary, summary_json_path)

    if SAVE_FRAME_TIME:
        np.save(output_dir / "frame_time.npy", np.array(frame_times, dtype=float))

    if SAVE_DETAILED_NUMPY:
        np.save(output_dir / "usage_samples.npy", usage_samples, allow_pickle=True)
        np.save(output_dir / "latency_stats.npy", latency_stats, allow_pickle=True)
        np.save(output_dir / "usage_summary.npy", usage_summary, allow_pickle=True)
        np.save(output_dir / "torch_peaks.npy", torch_peaks, allow_pickle=True)

    print("\n=== Performance Summary ===")
    print(f"Tracker: {tracker_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Total frames: {len(image_files)}")
    print(f"Execution time: {total_time:.2f} sec")
    print(f"Throughput FPS: {throughput_fps:.2f}" if throughput_fps is not None else "Throughput FPS: None")
    print(
        f"Latency mean/std: {latency_stats['mean_sec']:.4f} / {latency_stats['std_sec']:.4f} sec"
        if latency_stats["mean_sec"] is not None else "Latency: None"
    )
    print(
        f"CPU avg util: {usage_summary['cpu']['avg_util_percent']:.1f}%"
        if usage_summary["cpu"]["avg_util_percent"] is not None else "CPU avg util: None"
    )
    if usage_summary["gpu"]["avg_util_percent"] is not None:
        print(
            f"GPU util avg/max: "
            f"{usage_summary['gpu']['avg_util_percent']:.1f}% / "
            f"{usage_summary['gpu']['max_util_percent']:.1f}%"
        )
    else:
        print("GPU util: None")
    print(f"Tracked IDs after filtering: {len(track_id_boxes)}")
    print("===========================\n")


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    model = YOLO(MODEL_PATH)
    if TORCH_AVAILABLE:
        model.to(device)

    gpu_handle = init_nvml()

    try:
        for tracker_name, tracker_yaml in TRACKER_CONFIGS.items():
            for dataset_name in DATASETS:
                run_single_tracker_dataset(
                    model=model,
                    tracker_name=tracker_name,
                    tracker_yaml=tracker_yaml,
                    dataset_name=dataset_name,
                    gpu_handle=gpu_handle,
                )
    finally:
        shutdown_nvml()


if __name__ == "__main__":
    main()