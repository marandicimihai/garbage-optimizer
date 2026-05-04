import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from flask import Flask, Response, render_template, jsonify, request
import threading
import time
import requests as http_requests

app = Flask(__name__)

# ── Class names (same order as training) ────────────────────────────────────
CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]

# ── Smart-bin angle map ───────────────────────────────────────────────────────
ANGLE_MAP = {
    "plastic":   144,
    "metal":     288,
    "paper":     432,
    "cardboard": 432,
    "glass":     576,
}
DEFAULT_ANGLE = 0          # battery, biological, clothes, shoes, trash → close lid

BIN_BASE_URL = "http://192.168.99.117/move?angle={}"

# ── Model (loaded once at startup) ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_transform = None


def _build_model(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=None)
    num_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m


def _load_model() -> None:
    global _model, _transform
    model_path = Path(__file__).resolve().parents[2] / "models" / "best_resnet50.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    m = _build_model(len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    _model = m
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("Model loaded.")


# ── Webcam capture thread ─────────────────────────────────────────────────────
_camera_lock = threading.Lock()
_latest_frame = None
_camera_running = False


def _camera_loop() -> None:
    global _latest_frame, _camera_running
    cap = cv2.VideoCapture(0)
    # Give macOS camera permission / hardware time to warm up
    time.sleep(1.5)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0")
        _camera_running = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _camera_running = True
    print("Camera 0 opened successfully.")
    while _camera_running:
        ok, frame = cap.read()
        if ok:
            with _camera_lock:
                _latest_frame = frame.copy()
        else:
            time.sleep(0.01)
    cap.release()


def _generate_mjpeg():
    while True:
        with _camera_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes()
               + b"\r\n")
        time.sleep(0.033)   # ~30 fps cap


# ── Bin state (prevents overlapping sequences) ────────────────────────────────
_bin_busy = False
_bin_lock = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/classify", methods=["POST"])
def classify():
    with _camera_lock:
        frame = _latest_frame.copy() if _latest_frame is not None else None
    if frame is None:
        return jsonify(error="Camera not ready"), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = _transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)

    label = CLASS_NAMES[int(idx.item())]
    angle = ANGLE_MAP.get(label, DEFAULT_ANGLE)

    return jsonify(
        label=label,
        confidence=round(float(confidence.item()) * 100, 1),
        angle=angle,
    )


@app.route("/send_to_bin", methods=["POST"])
def send_to_bin():
    global _bin_busy
    with _bin_lock:
        if _bin_busy:
            return jsonify(error="Bin is busy"), 409
        _bin_busy = True

    data = request.get_json(silent=True) or {}
    angle = int(data.get("angle", DEFAULT_ANGLE))

    try:
        # Move bin to target position
        http_requests.get(BIN_BASE_URL.format(angle), timeout=5)
        # Wait 7 seconds (lid open)
        time.sleep(7)
        # Return to closed position
        http_requests.get(BIN_BASE_URL.format(0), timeout=5)
    except Exception as exc:
        print(f"Bin request failed: {exc}")
    finally:
        with _bin_lock:
            _bin_busy = False

    return jsonify(ok=True)


if __name__ == "__main__":
    _load_model()
    cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    cam_thread.start()
    print("Starting server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
