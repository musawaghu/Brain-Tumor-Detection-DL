import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import models

# Make the project root importable when running `pytest` from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import split_data  # noqa: E402
from gradcam import grad_cam, overlay  # noqa: E402


def test_split_is_disjoint_complete_and_stratified():
    df = pd.read_csv(os.path.join(ROOT, "brain_tumor_labels.csv"))
    train_df, val_df, test_df = split_data(df)

    train, val, test = set(train_df.filename), set(val_df.filename), set(test_df.filename)
    # No image appears in more than one split
    assert not (train & val) and not (train & test) and not (val & test)
    # Every image is used exactly once
    assert len(train) + len(val) + len(test) == len(df)
    # Roughly 70 / 15 / 15
    assert abs(len(test_df) / len(df) - 0.15) < 0.01
    # Each split keeps the overall tumor ratio (within 2 percentage points)
    overall = df.label.mean()
    for split in (train_df, val_df, test_df):
        assert abs(split.label.mean() - overall) < 0.02


def test_split_is_reproducible():
    df = pd.read_csv(os.path.join(ROOT, "brain_tumor_labels.csv"))
    assert list(split_data(df)[2].filename) == list(split_data(df)[2].filename)


def test_grad_cam_shape_range_and_hook_cleanup():
    model = models.resnet50(weights=None, num_classes=2)
    cam = grad_cam(model, torch.randn(1, 3, 224, 224), target_class=1)

    assert cam.shape == (224, 224)
    assert cam.min() >= 0 and cam.max() <= 1
    # The forward hook must be removed afterwards
    assert len(model.layer4._forward_hooks) == 0


def test_overlay_returns_rgb_image_of_cam_size():
    img = Image.fromarray(np.random.randint(0, 255, (300, 400), dtype=np.uint8))  # grayscale input
    out = overlay(img, np.random.rand(224, 224))
    assert out.size == (224, 224) and out.mode == "RGB"


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "final_brain_mri_model.pth")),
                    reason="trained weights not present (run main.py first)")
def test_app_predict_on_sample():
    os.chdir(ROOT)
    import app

    sample = sorted(os.listdir("samples"))[0]
    confidences, heatmap = app.predict(Image.open(os.path.join("samples", sample)))

    assert set(confidences) == set(app.LABELS)
    assert abs(sum(confidences.values()) - 1) < 1e-5
    assert heatmap.size == (224, 224)
