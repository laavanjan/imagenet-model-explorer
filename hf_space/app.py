"""Gradio front-end for the Hugging Face Space: one accurate, CPU-friendly model + Grad-CAM."""
from __future__ import annotations

import gradio as gr
from PIL import Image, ImageOps

from classifier import ImageClassifier, confidence_level, load_image_from_url, overlay_heatmap

try:  # ZeroGPU Spaces require at least one @spaces.GPU function; elsewhere this is a no-op.
    import spaces

    gpu = spaces.GPU
except ImportError:

    def gpu(fn):
        return fn


MODEL_KEY = "convnext_tiny"  # 82.5 % top-1, 28 M parameters, ~110 MB: accurate yet fast on CPU
clf = ImageClassifier(MODEL_KEY)  # weights are downloaded once, at Space start-up

VERDICT = {
    "high": "✅ **High confidence** - the model is quite sure.",
    "moderate": "🟡 **Moderate confidence** - probably right, but check the runner-ups.",
    "uncertain": "🟠 **Uncertain** - the top guesses are close; treat the label with care.",
    "unreliable": "🔴 **Unreliable** - the model is guessing. The object is probably not one of the "
    "1,000 ImageNet categories, so the label is only the *nearest* match.",
}


def _to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert("RGB")


def classify_url(url: str | None):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste an image link (starting with http:// or https://) first.")
    try:
        img = load_image_from_url(url)  # fetched on CPU, outside the GPU slot
    except Exception as exc:
        raise gr.Error(f"Couldn't load an image from that link: {exc}") from exc
    return classify(img)


def classify(img: Image.Image | None):
    if img is None:
        raise gr.Error("Please upload an image first.")
    return _predict(_to_rgb(img))


@gpu
def _predict(img: Image.Image):
    pred = clf.predict(img, top_k=5)
    cam = clf.grad_cam(img, pred.indices[0])
    heat = overlay_heatmap(clf.model_view(img), cam)
    scores = dict(zip(pred.labels, pred.probs))
    note = f"{VERDICT[confidence_level(pred)]}\n\nInference time: {pred.latency_ms:.0f} ms"
    return scores, heat, note


with gr.Blocks(title="Image Classifier") as demo:
    gr.Markdown(
        "# 🔬 Image Classifier\n"
        "Upload a photo. A pretrained **ConvNeXt-Tiny** names what it sees, says how sure it is, "
        "and shows (Grad-CAM) where in the picture it looked."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Your image", sources=["upload", "webcam", "clipboard"])
            btn = gr.Button("Classify", variant="primary")
            url_in = gr.Textbox(label="...or paste an image link", placeholder="https://example.com/photo.jpg")
            url_btn = gr.Button("Classify link")
        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="Top 5 predictions")
            out_note = gr.Markdown()
            out_heat = gr.Image(label="Grad-CAM: red/yellow = what drove the top answer")
    btn.click(classify, inp, [out_label, out_heat, out_note])
    inp.upload(classify, inp, [out_label, out_heat, out_note])
    url_btn.click(classify_url, url_in, [out_label, out_heat, out_note])
    url_in.submit(classify_url, url_in, [out_label, out_heat, out_note])

if __name__ == "__main__":
    demo.launch()
