import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from collections import deque


DEFAULT_CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(model_path: Path, device: torch.device, class_names: list[str]) -> nn.Module:
    model = build_model(len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def predict_frame(model: nn.Module, frame_bgr: np.ndarray, device: torch.device, transform, class_names: list[str]):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[int(predicted_idx.item())]
    return predicted_class, float(confidence.item())


def draw_overlay(frame: np.ndarray, label: str, confidence: float, fps: float) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2%}   FPS: {fps:.1f}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "best_resnet50.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, DEFAULT_CLASS_NAMES)
    transform = make_transform(224)

    camera_idx = 0
    print(f"Using camera {camera_idx}")

    capture = cv2.VideoCapture(camera_idx)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_idx}")

    recent_predictions: deque[str] = deque(maxlen=7)
    recent_confidences: deque[float] = deque(maxlen=7)

    previous_time = cv2.getTickCount()
    ticks_per_second = cv2.getTickFrequency()

    print("Press q to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            label, confidence = predict_frame(model, frame, device, transform, DEFAULT_CLASS_NAMES)
            recent_predictions.append(label)
            recent_confidences.append(confidence)

            display_label = max(set(recent_predictions), key=recent_predictions.count)
            display_confidence = float(np.mean(recent_confidences))

            current_time = cv2.getTickCount()
            fps = ticks_per_second / max(current_time - previous_time, 1)
            previous_time = current_time

            annotated_frame = draw_overlay(frame, display_label, display_confidence, fps)
            cv2.imshow("Garbage Classifier", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()