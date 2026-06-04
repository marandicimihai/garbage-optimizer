# Garbage Optimizer

Project Overview
----------------

Garbage Optimizer is a small smart waste management system in two main phases:

- Phase 1 — Bin live detection (camera + controlled compartments): a ResNet50-based classifier detects the waste type from a webcam feed and the system opens only the compartment corresponding to the detected material, preventing incorrect disposal.
- Phase 2 — Bin health & prioritization (data-driven): scripts generate simulated disposal events and compute per-bin health predictions (urgency, overflow risk, expected collected weight). The visualization surfaces per-bin urgency so operators can prioritize pickups without requiring a full route optimizer.

See the `src/garbage-classifier` and `src/route-optimizer` folders for implementation.

Problem Description
-------------------

The project enforces correct recycling by combining a camera-based classifier with hardware control (bin lid/compartment angles) and then uses the recorded events to optimize waste collection routes. This reduces sorting errors at source and enables more efficient, data-driven collection schedules.

Data Sources
------------

- Image data source: Garbage Classification v2 (Kaggle) — https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2 (see `data/data.md`).
- The route optimizer uses synthetic/derived data produced by the project's scripts in `src/route-optimizer/generated/` (POIs, street lines, generated bins, and waste_events.csv). Run `python src/route-optimizer/setup.py` to generate these files.

Dataset used for training
-------------------------

The classifier is trained on a cleaned 4-class dataset stored in `data/dataset/archive/merged_4class_256/`.

Source folders used to build it:

- `data/dataset/archive/standardized_256/` for the original archive classes already resized to 256x256.
- `data/dataset-resized/` for the newer resized images.

Merge rules:

- Keep only `glass`, `metal`, `paper`, and `plastic`.
- Map `cardboard` into `paper`.
- Drop `battery`, `biological`, `clothes`, `shoes`, and `trash`.
- Resize everything to 256x256 before training.

Final training set statistics:

| Class | Images | Size |
| --- | ---: | --- |
| glass | 2237 | 256x256 |
| metal | 1340 | 256x256 |
| paper | 3744 | 256x256 |
| plastic | 2079 | 256x256 |

The notebooks in `notebooks/` point to this merged dataset.

How to run
----------

Prerequisites

- Python 3.9+ (3.10 recommended)
- A CUDA-capable GPU is optional for faster model inference; CPU also works.

Recommended Python packages (install with pip):

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy opencv-python flask scikit-learn
```

Note: use an environment manager (venv/conda). If you have a local `requirements.txt`, install from it instead.

Run the physical bin prototype (web server + camera):

```bash
python src/garbage-classifier/app.py
```

This starts a Flask server (default port 5001) serving a live MJPEG feed and endpoints for classification and sending bin angle commands. The script expects a checkpoint at `models/best_resnet50.pth`.

Alternatively, run the local webcam demo (no Flask server):

```bash
python src/garbage-classifier/live_webcam.py
```

Run the route optimizer and start the visualization web server (the `setup.py` script runs all generator scripts, then launches the server):

```bash
python src/route-optimizer/setup.py
```

This runs the following scripts in order (they write outputs into `src/route-optimizer/generated/`):

- `pois.py` — loads or synthesizes points of interest (POIs)
- `streets.py` — prepares street line data
- `bins.py` — places candidate bins along street lines
- `waste_data.py` — synthesizes waste events using POIs, clustered ambient background waste, and random spill events across bins

Then it starts `server.py` and opens `http://127.0.0.1:5002` in your browser. The server computes per-bin health predictions (urgency, overflow estimates) and exposes them via the API.

You can also run only the server (if CSV files are already generated):

```bash
python src/route-optimizer/server.py
```

Files produced
--------------

- `src/route-optimizer/generated/waste_events.csv` — simulated disposal events
- `src/route-optimizer/generated/bins.csv` — computed bin locations
- Per-bin `bin_health` predictions are available from the server API (`/api/status`) and surfaced in the map UI.

Library versions (recommended)
-----------------------------

These are the packages used by the code; install compatible versions for best results:

- Python: 3.9 or 3.10
- torch: >=1.12 (or newer matching your CUDA, e.g., 2.x)
- torchvision: compatible with your torch version
- numpy: >=1.21
- opencv-python: >=4.5
- flask: >=2.0
- scikit-learn: >=1.0

If you run into compatibility issues with `torch`/`torchvision`, pick versions that match your Python and CUDA setup per the official PyTorch selector.

Screenshots & Results
---------------------

![Visualization screenshot](docs/Screenshot%202026-05-05%20at%2022.25.30.png)

Prototype video (physical bin)
------------------------------

Video link: https://youtu.be/szBxeMfS7wU

Notes for judges / reproducibility
--------------------------------

- The visualization server can be started by running `python src/route-optimizer/setup.py` (this runs the generator scripts and opens `http://127.0.0.1:5002`).
- The classifier expects a trained checkpoint at `models/best_resnet50.pth`. Training notebooks are in `notebooks/`.
- Example generated inputs are included in `src/route-optimizer/generated/` so the visualization can be opened without re-running the pipeline.

Project structure (relevant files)
----------------------------------

- `src/garbage-classifier/app.py` — Flask server for the physical prototype (camera feed, `/classify`, `/send_to_bin` endpoints)
- `src/garbage-classifier/live_webcam.py` — local webcam demo for classifier
- `models/best_resnet50.pth` — trained classifier checkpoint used by the demos
- `src/route-optimizer/*.py` — scripts producing POIs, street lines, bins, waste events, and visualization (including per-bin health predictions)
- `src/route-optimizer/server.py` — Flask server for visualization and per-bin health predictions
- `src/route-optimizer/generated/` — generated CSVs used by the visualization server
