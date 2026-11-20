"""Tests for the inference engine. Run: ``python -m pytest``.

* Every architecture is exercised with random weights (no downloads) to prove that
  prediction, Grad-CAM and pre-processing are wired up correctly.
* One test loads the real pretrained ResNet-50 (cached locally after the first run) and checks
  an actual prediction.
"""
import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from classifier import (
    COLORMAP_NAMES,
    MODEL_SPECS,
    ImageClassifier,
    Prediction,
    confidence_level,
    heatmap_only,
    load_image_from_bytes,
    load_image_from_url,
    overlay_heatmap,
)


def synthetic() -> Image.Image:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    ImageDraw.Draw(img).ellipse((120, 60, 280, 240), fill=(220, 30, 30))
    return img


# --------------------------------------------------------------------- every architecture
@pytest.fixture(scope="module", params=list(MODEL_SPECS))
def untrained(request) -> ImageClassifier:
    torch.manual_seed(0)  # deterministic random init -> reproducible tests
    return ImageClassifier(request.param, device="cpu", pretrained=False)


def test_entropy_is_clamped_to_unit_interval():
    """Regression: a near-uniform output used to give entropy 1.0000003 (seed 18 of EfficientNet-V2-S)."""
    torch.manual_seed(18)
    clf = ImageClassifier("efficientnet_v2_s", device="cpu", pretrained=False)
    assert 0 <= clf.predict(synthetic(), top_k=5).entropy <= 1


def test_predict_is_a_valid_distribution(untrained):
    img = synthetic()
    pred = untrained.predict(img, top_k=5)
    assert len(pred.labels) == 5
    assert pred.probs == sorted(pred.probs, reverse=True)
    assert abs(sum(untrained.predict(img, top_k=1000).probs) - 1) < 1e-3
    assert 0 <= pred.entropy <= 1
    assert len(untrained.categories) == 1000


def test_tta_runs(untrained):
    assert len(untrained.predict(synthetic(), top_k=3, tta=True).labels) == 3


def test_grad_cam_shape_and_range(untrained):
    cam = untrained.grad_cam(synthetic(), class_idx=0)
    n = untrained.input_size
    assert cam.shape == (n, n)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_grad_cam_leaves_no_hooks_behind(untrained):
    layer = untrained.spec.target_layer(untrained.model)
    untrained.grad_cam(synthetic(), class_idx=1)
    assert len(layer._forward_hooks) == 0


def test_model_view_matches_input_size(untrained):
    assert untrained.model_view(synthetic()).size == (untrained.input_size, untrained.input_size)


# ----------------------------------------------------------------------- real weights
@pytest.fixture(scope="module")
def resnet50() -> ImageClassifier:
    return ImageClassifier("resnet50", device="cpu")


def test_pretrained_resnet50_sanity(resnet50):
    assert resnet50.n_params == pytest.approx(25.6e6, rel=0.01)
    # A flat grey image has no object in it, so the model must not be confident about anything.
    pred = resnet50.predict(Image.new("RGB", (256, 256), (128, 128, 128)), top_k=3)
    assert pred.probs[0] < 0.5


def test_pretrained_resnet50_is_deterministic(resnet50):
    img = synthetic()
    a, b = resnet50.predict(img, top_k=5), resnet50.predict(img, top_k=5)
    assert a.indices == b.indices and np.allclose(a.probs, b.probs)


# ----------------------------------------------------------------------- confidence bands
@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.95, "high"), (0.70, "high"), (0.50, "moderate"), (0.21, "uncertain"), (0.088, "unreliable")],
)
def test_confidence_level(p, expected):
    pred = Prediction([0], ["x"], [p], 1.0, 0.5)
    assert confidence_level(pred) == expected


def test_unknown_model_key_is_rejected():
    with pytest.raises(KeyError):
        ImageClassifier("not-a-model", pretrained=False)


# --------------------------------------------------------------------------- rendering
@pytest.mark.parametrize("cmap", COLORMAP_NAMES)
def test_colormaps_render(cmap):
    cam = np.random.default_rng(1).random((224, 224)).astype(np.float32)
    base = synthetic().resize((224, 224))
    assert overlay_heatmap(base, cam, 0.5, cmap).size == base.size
    assert heatmap_only(cam, (100, 80), cmap).size == (100, 80)


# --------------------------------------------------------------------------- image I/O
def test_rgba_image_is_flattened_to_rgb():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (255, 0, 0, 0)).save(buf, format="PNG")
    img = load_image_from_bytes(buf.getvalue())
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (60, 30), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert load_image_from_bytes(buf.getvalue()).size == (30, 60)


def test_garbage_bytes_raise():
    with pytest.raises(Exception):  # noqa: B017  (PIL raises UnidentifiedImageError)
        load_image_from_bytes(b"definitely not an image")


@pytest.mark.parametrize("bad", ["ftp://x/y.jpg", "file:///etc/passwd", "not a url", "javascript:alert(1)"])
def test_non_http_urls_are_rejected(bad):
    with pytest.raises(ValueError):
        load_image_from_url(bad)
