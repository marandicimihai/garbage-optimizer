#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms


def list_cameras():
    """List all available cameras on the system."""
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def load_checkpoint(checkpoint_path, device):
    data = torch.load(checkpoint_path, map_location=device)
    class_names = data.get("class_names") if isinstance(data, dict) else None
    state_dict = data.get("model_state") if isinstance(data, dict) else data
    if state_dict is None:
        raise RuntimeError(f"No model state found in {checkpoint_path}")
    return state_dict, class_names


def build_model(num_classes, device, state_dict=None):
    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.last_channel, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes),
    )
    if state_dict is not None:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def predict_frame(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, index = torch.max(probs, dim=1)
    return index.item(), float(confidence.item())


def main():
    parser = argparse.ArgumentParser(description="Live webcam classification")
    script_dir = Path(__file__).parent
    default_checkpoint = script_dir / "garbage_model.pth"
    parser.add_argument("--checkpoint", "-c", type=Path, default=default_checkpoint, help="Path to model checkpoint")
    parser.add_argument("--camera", "-k", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--device", "-d", default=None, help="torch device (cpu, mps, cuda). Auto-detected by default")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    # List cameras if requested
    available = list_cameras()
    if args.list_cameras:
        print(f"Available cameras: {available}")
        return
    
    print(f"Using camera index: {args.camera}")

    device = torch.device(args.device) if args.device else torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    state_dict, class_names = load_checkpoint(args.checkpoint, device)
    if class_names is None:
        raise RuntimeError("Checkpoint does not include `class_names`. Save file must contain class_names list.")

    model = build_model(len(class_names), device, state_dict=state_dict)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame)
        index, confidence = predict_frame(model, device, input_tensor)
        label = class_names[index]

        text = f"{label}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)

        now = time.time()
        fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Live Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
