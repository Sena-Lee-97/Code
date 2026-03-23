# DEP-Track: Deep Learning-Based Single-Cell Tracking for Crossover Frequency Estimation in Dielectrophoresis

---

## 📌 Overview

This repository provides the official implementation of **DEP-Track**, a computational framework for automated single-cell trajectory extraction and crossover frequency estimation in dielectrophoresis (DEP) experiments.

Precise analysis of DEP responses at the single-cell level remains challenging, particularly in long-term imaging experiments with frequency modulation and dense cell populations. DEP-Track addresses this limitation by enabling:

* Robust long-term single-cell tracking (up to 13,200 frames)
* Motion-aware trajectory association under abrupt polarity transitions
* Automated estimation of crossover frequency at the single-cell level
* Scalable and reproducible analysis for large cell populations

---

## 🔬 Key Features

* Deep learning-based cell detection (YOLO)
* Motion-aware trajectory association (proposed method)
* Long-term identity preservation across frames
* Automated crossover frequency estimation
* Quantitative comparison with baseline trackers (Bot-SORT, ByteTrack, OC-SORT)

---

## 📁 Repository Structure

```id="repo-structure"
.
├── data/                      # Input image sequences
├── results/
│   ├── detection_result/      # Detection outputs
│   ├── tracking_result/       # Proposed method outputs
│   ├── baseline_tracking/     # Baseline tracker outputs
│
├── matlab/                    # DEP-Track (proposed method)
│   ├── run_cell_tracking.m
│   ├── main_tracking.m
│   ├── confirm_tracking.m
│   ├── save_cell_xy.m
│
├── python/                    # Baseline + visualization
│   ├── run_baseline_trackers.py
│   ├── postprocess_baseline_tracks.py
│   ├── save_tracking.py
│   ├── save_tracking_baseline.py
│
└── README.md
```

---

# 🚀 Execution Pipeline

## 1. Proposed Method (DEP-Track)

### Step 1. Run tracking (MATLAB)

```matlab id="run-matlab"
run_cell_tracking
```

This step performs:

* Loading detection results
* Motion-aware trajectory tracking
* Out-of-bound and incomplete trajectory filtering
* Generation of final tracking results

Output:

```id="proposed-output"
results/tracking_result/<dataset>/save_files/
└── <dataset>cell_xy.csv
```

---

### Step 2. Generate visualization (Python)

```bash id="run-vis"
python save_tracking.py
```

This step:

* Loads tracking results (CSV)
* Overlays trajectories on images
* Generates video output (with configurable frame interval)

Output:

```id="video-output"
results/tracking_result/<dataset>/save_track/
└── tracking_overlay_step10.mp4
```

---

## 2. Baseline Methods (Bot-SORT, ByteTrack, OC-SORT)

### Step 1. Run baseline trackers

```bash id="run-baseline"
python run_baseline_trackers.py
```

This step:

* Applies YOLO-based detection and tracking
* Runs multiple trackers under identical conditions

Output:

```id="baseline-output"
results/baseline_tracking/<tracker>/<dataset>/
└── <dataset>.txt
```

Tracking format:

```id="tracking-format"
track_id, frame, x, y, width, height
```

---

### Step 2. Post-processing

```bash id="postprocess"
python postprocess_baseline_tracks.py
```

This step:

* Removes out-of-bound cells
* Filters incomplete trajectories
* Re-labels tracks
* Generates statistical plots

Outputs:

```id="postprocess-output"
track_cell_info.pkl
new_save_label.pkl
survival_curve_cells.png
tracking_length_histogram.png
```

---

### Step 3. Visualization

```bash id="baseline-vis"
python save_tracking_baseline.py
```

Output:

```id="baseline-video"
tracking_overlay.mp4
```

---

# 📊 Dataset Format

Input images should follow:

```id="dataset-format"
data/<dataset>/
├── IMG_0.jpg
├── IMG_1.jpg
├── ...
```

Requirements:

* Sequential frame indices
* Consistent resolution
* Naming format: `IMG_<index>.jpg`

---

# ⚖️ Experimental Setup

To ensure fair comparison:

* Identical dataset is used across all methods
* The same detection model (YOLO) is applied
* Only the tracking algorithm differs
* All hyperparameters for detection remain fixed

---

# 📈 Performance Metrics

The following metrics are computed:

* Tracking time
* Frames per second (FPS)
* CPU utilization
* Memory consumption
* Tracking length distribution
* Survival curve (tracking persistence)

---

# 🔁 Reproducibility

## Proposed Method

```id="reproduce-proposed"
run_cell_tracking
→ save_tracking.py
```

## Baseline Methods

```id="reproduce-baseline"
run_baseline_trackers.py
→ postprocess_baseline_tracks.py
→ save_tracking_baseline.py
```

---

# 📦 Code Availability

All code used in this study is publicly available to ensure full reproducibility of the experimental results.

---

# 📄 Citation

If you use this code, please cite:

```id="citation"
Lee et al., DEP-Track: Deep Learning-Based Single-Cell Tracking for Crossover Frequency Estimation in Dielectrophoresis
```

---

# 📬 Contact

Sejung Yang
Department of Precision Medicine
Yonsei University Wonju College of Medicine

📧 [syang@yonsei.ac.kr](mailto:syang@yonsei.ac.kr)

---

# 📝 License

(To be specified)
