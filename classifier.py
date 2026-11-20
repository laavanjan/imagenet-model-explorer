"""Core inference engine: pretrained ImageNet classifiers + Grad-CAM, independent of Streamlit.

Several torchvision architectures are exposed through one interface (see ``MODEL_SPECS``).
Each spec knows how to build the network, which pretrained weights to use and which layer
Grad-CAM should hook, so adding a model is a one-line change.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    convnext_small,
    convnext_tiny,
    efficientnet_v2_s,
    resnet50,
    resnet101,
)

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024

# Anchor colours (RGB 0-255) for a few hand-rolled colormaps, so we don't need matplotlib.
_COLORMAPS: dict[str, list[tuple[int, int, int]]] = {
    "Jet": [(0, 0, 131), (0, 60, 255), (0, 255, 255), (255, 255, 0), (255, 60, 0), (128, 0, 0)],
    "Inferno": [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
    "Viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "Magma": [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
}
COLORMAP_NAMES = list(_COLORMAPS)


# ------------------------------------------------------------------------------ model zoo
@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one classifier and explain it with Grad-CAM."""

    key: str
    name: str
    builder: Callable[..., nn.Module]
    weights: Any  # a torchvision ``*_Weights`` enum member
    target_layer: Callable[[nn.Module], nn.Module]  # last conv stage: where Grad-CAM hooks in

    @property
    def acc1(self) -> float:
        """Published ImageNet-1k top-1 accuracy (%) of these weights, straight from torchvision."""
        return float(self.weights.meta["_metrics"]["ImageNet-1K"]["acc@1"])

    @property
    def size_mb(self) -> float:
        return float(self.weights.meta["_file_size"])

    @property
    def params_m(self) -> float:
        return float(self.weights.meta["num_params"]) / 1e6


MODEL_SPECS: dict[str, ModelSpec] = {
    s.key: s
    for s in (
        ModelSpec("resnet50", "ResNet-50", resnet50, ResNet50_Weights.IMAGENET1K_V2, lambda m: m.layer4[-1]),
        ModelSpec("resnet101", "ResNet-101", resnet101, ResNet101_Weights.IMAGENET1K_V2, lambda m: m.layer4[-1]),
        ModelSpec("convnext_tiny", "ConvNeXt-Tiny", convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, lambda m: m.features[-1]),
        ModelSpec("convnext_small", "ConvNeXt-Small", convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, lambda m: m.features[-1]),
        ModelSpec("efficientnet_v2_s", "EfficientNet-V2-S", efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1, lambda m: m.features[-1]),
    )
}
DEFAULT_MODEL = "resnet50"


@dataclass
class Prediction:
    indices: list[int]
    labels: list[str]
    probs: list[float]
    latency_ms: float
    entropy: float  # normalised 0..1 (0 = certain, 1 = uniform)


# Confidence bands on the top-1 probability. Below ``UNRELIABLE_BELOW`` the image most likely
# shows something outside the 1,000 ImageNet classes (e.g. a hibiscus), so the label is only
# the *nearest* category, not an answer.
HIGH_ABOVE = 0.70
MODERATE_ABOVE = 0.35
UNRELIABLE_BELOW = 0.15


def confidence_level(pred: Prediction) -> str:
    """Bucket a prediction into 'high', 'moderate', 'uncertain' or 'unreliable'."""
    p = pred.probs[0]
    if p >= HIGH_ABOVE:
        return "high"
    if p >= MODERATE_ABOVE:
        return "moderate"
    if p >= UNRELIABLE_BELOW:
        return "uncertain"
    return "unreliable"


# ------------------------------------------------------------------------------- image I/O
def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode bytes to an upright RGB image (handles EXIF rotation and transparency)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    headers = {"User-Agent": "vision-lab/2.0"}
    with requests.get(url, headers=headers, timeout=15, stream=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if ctype and not ctype.lower().startswith("image/"):
            raise ValueError(f"URL did not return an image (Content-Type: {ctype}).")
        buf = bytearray()
        for chunk in r.iter_content(64 * 1024):
            buf.extend(chunk)
            if len(buf) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Image is larger than 15 MB.")
    return load_image_from_bytes(bytes(buf))


# ------------------------------------------------------------------------------ rendering
def apply_colormap(cam: np.ndarray, name: str = "Jet") -> np.ndarray:
    """Map a [0,1] float array of shape (H, W) to an (H, W, 3) uint8 RGB array."""
    anchors = np.array(_COLORMAPS[name], dtype=np.float32)
    pos = np.linspace(0.0, 1.0, len(anchors))
    flat = np.clip(cam, 0, 1).ravel()
    rgb = np.stack([np.interp(flat, pos, anchors[:, c]) for c in range(3)], axis=-1)
    return rgb.reshape(*cam.shape, 3).astype(np.uint8)


def overlay_heatmap(base: Image.Image, cam: np.ndarray, alpha: float = 0.5, cmap: str = "Jet") -> Image.Image:
    """Blend a Grad-CAM map onto an image."""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = apply_colormap(cam_resized, cmap).astype(np.float32)
    # Weight blend by activation so cold regions stay close to the original photo.
    w = alpha * np.clip(cam_resized * 1.5, 0, 1)[..., None]
    out = np.asarray(base, dtype=np.float32) * (1 - w) + heat * w
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def heatmap_only(cam: np.ndarray, size: tuple[int, int], cmap: str = "Jet") -> Image.Image:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    return Image.fromarray(apply_colormap(np.asarray(cam_img, dtype=np.float32) / 255.0, cmap))


# ------------------------------------------------------------------------------ classifier
class ImageClassifier:
    """ImageNet-1k classifier with top-k prediction and Grad-CAM explanations.

    ``pretrained=False`` builds the same architecture with random weights and skips the download;
    it is used by the architecture tests and the latency benchmark.
    """

    def __init__(self, key: str = DEFAULT_MODEL, device: str | None = None, pretrained: bool = True):
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model '{key}'. Choose from: {', '.join(MODEL_SPECS)}")
        self.spec = MODEL_SPECS[key]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.spec.builder(weights=self.spec.weights if pretrained else None).eval().to(self.device)
        self.categories: list[str] = list(self.spec.weights.meta["categories"])
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # Pre-processing recipe that matches the weights (resize / crop sizes differ per model).
        recipe = self.spec.weights.transforms()
        self.input_size = int(recipe.crop_size[0])
        self.view = T.Compose(
            [T.Resize(int(recipe.resize_size[0]), interpolation=recipe.interpolation), T.CenterCrop(self.input_size)]
        )
        self.norm = T.Compose([T.ToTensor(), T.Normalize(recipe.mean, recipe.std)])

    # ------------------------------------------------------------------ helpers
    def model_view(self, img: Image.Image) -> Image.Image:
        """What the network actually sees: the resized, centre-cropped input."""
        return self.view(img)

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        return self.norm(self.view(img)).unsqueeze(0).to(self.device)

    # --------------------------------------------------------------- prediction
    @torch.inference_mode()
    def predict(self, img: Image.Image, top_k: int = 5, tta: bool = False) -> Prediction:
        t0 = time.perf_counter()
        x = self._tensor(img)
        probs = F.softmax(self.model(x), dim=1)
        if tta:
            probs = (probs + F.softmax(self.model(torch.flip(x, dims=[3])), dim=1)) / 2
        p = probs[0].cpu()
        latency = (time.perf_counter() - t0) * 1000
        top = torch.topk(p, top_k)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum().item() / float(np.log(len(p)))
        ent = min(max(ent, 0.0), 1.0)  # float rounding can push a near-uniform output to 1.0000003
        idx = top.indices.tolist()
        return Prediction(idx, [self.categories[i] for i in idx], top.values.tolist(), latency, ent)

    # ----------------------------------------------------------------- Grad-CAM
    def grad_cam(self, img: Image.Image, class_idx: int) -> np.ndarray:
        """Grad-CAM for ``class_idx``, upsampled to the model input size and scaled to [0, 1]."""
        store: dict[str, torch.Tensor] = {}
        layer = self.spec.target_layer(self.model)

        def fwd_hook(_m, _i, out):
            store["act"] = out.detach()
            out.register_hook(lambda g: store.__setitem__("grad", g.detach()))

        handle = layer.register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                x = self._tensor(img).requires_grad_(True)
                logits = self.model(x)
                self.model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
        finally:
            handle.remove()

        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()


class ResNet50Classifier(ImageClassifier):
    """Backwards-compatible alias for the original single-model API."""

    def __init__(self, device: str | None = None, pretrained: bool = True):
        super().__init__("resnet50", device=device, pretrained=pretrained)
