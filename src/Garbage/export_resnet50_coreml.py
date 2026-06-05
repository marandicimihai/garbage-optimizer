from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = [
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    # Accept either models/best_resnet50.pth (original name) or models/model.pth (new)
    checkpoint_path = repo_root / "models" / "best_resnet50.pth"
    alt_path = repo_root / "models" / "model.pth"
    if not checkpoint_path.exists() and alt_path.exists():
        checkpoint_path = alt_path
    output_path = repo_root / "src" / "Garbage" / "Garbage" / "ResNet.mlpackage"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(len(CLASS_NAMES))
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                shape=example_input.shape,
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
            )
        ],
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.save(output_path)
    print(f"Saved CoreML model to {output_path}")


if __name__ == "__main__":
    main()