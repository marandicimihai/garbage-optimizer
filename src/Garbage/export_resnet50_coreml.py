from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = [
    "glass",
    "metal",
    "paper",
    "plastic",
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
    # Prefer models/model.pth if present (new 4-class checkpoint); fall back to best_resnet50.pth
    model_pth = repo_root / "models" / "model.pth"
    best_pth = repo_root / "models" / "best_resnet50.pth"
    if model_pth.exists():
        checkpoint_path = model_pth
    else:
        checkpoint_path = best_pth
    output_path = repo_root / "src" / "Garbage" / "Garbage" / "ResNet.mlpackage"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch versions don't accept weights_only
        loaded = torch.load(checkpoint_path, map_location="cpu")

    # The checkpoint may be a full nn.Module object or a state_dict/dict wrapper.
    if isinstance(loaded, dict):
        # extract nested state_dict if present
        state_dict = None
        for key in ("model_state", "state_dict", "state"):
            if key in loaded:
                state_dict = loaded[key]
                break
        if state_dict is None:
            state_dict = loaded

        model = build_model(len(CLASS_NAMES))
        model.load_state_dict(state_dict)
    else:
        # assume it's a serialized nn.Module
        model = loaded

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