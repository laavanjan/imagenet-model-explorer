"""Gradio front-end for the Hugging Face Space: one accurate, CPU-friendly model + Grad-CAM."""
from __future__ import annotations

import html
import os

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

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [  # (file, label); photos are CC0 / public domain from scikit-image's sample data
    ("cat.jpg", "🐱 Cat"),
    ("coffee.jpg", "☕ Coffee"),
    ("abstract.jpg", "🎨 Abstract art"),
]
DEFAULT_STRENGTH = 0.55

# name, one-line advice, accent colour
LEVELS = {
    "high": ("High confidence", "The model is quite sure about this one.", "#16a34a"),
    "moderate": ("Moderate confidence", "Probably right - check the runner-ups below.", "#ca8a04"),
    "uncertain": ("Uncertain", "The top guesses are close, so treat the label with care.", "#ea580c"),
    "unreliable": (
        "Unreliable",
        "The model is guessing. This is probably not one of the 1,000 ImageNet categories, "
        "so the label is only the nearest match.",
        "#dc2626",
    ),
}

HERO = """
<div class="hero">
  <h1>🔬 Image Classifier</h1>
  <p>Drop in a photo. A neural network names what it sees, tells you how sure it is,
     and shows <b>where in the picture it looked</b>.</p>
  <div class="chips">
    <span>ConvNeXt-Tiny</span><span>82.5 % ImageNet top-1</span>
    <span>1,000 categories</span><span>Grad-CAM heatmap</span>
  </div>
</div>
"""

EMPTY_CARD = """
<div class="card placeholder">
  <div class="big">👈</div>
  <div><b>Add an image to get started</b></div>
  <div>Upload a photo, use your webcam, paste a link, or click one of the examples.</div>
</div>
"""

GUIDE = """
| Verdict | Top probability | What it means |
|---|---|---|
| **High confidence** | 70 % or more | The model is fairly sure. |
| **Moderate confidence** | 35 – 70 % | Probably right; check the runner-ups. |
| **Uncertain** | 15 – 35 % | Several answers are close; treat it as a guess. |
| **Unreliable** | under 15 % | Likely outside the 1,000 categories; the label is only the nearest match. |

**Heatmap (Grad-CAM):** red and yellow areas are the parts of the photo that pushed the model towards its top
answer. The model never sees your whole photo: it is resized and centre-cropped first, so anything outside the crop
(left picture) is ignored.

**Limits:** the model only knows the 1,000 ImageNet categories, and a confident answer is not always a correct one.
"""

FOOTER = """
<div class="footer">
  Model: ConvNeXt-Tiny (torchvision, trained on ImageNet-1k) · Example photos: scikit-image sample data (CC0 / public domain) ·
  <a href="https://github.com/laavanjan/imagenet-model-explorer" target="_blank" rel="noopener">Source code</a>
</div>
"""

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto; }
.hero { padding: 28px 28px 20px; border-radius: 20px; margin-bottom: 6px; color: #fff;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%); }
.hero, .hero * { color: #fff; }
.hero h1 { margin: 0 0 6px; font-size: 2.1rem; font-weight: 800; color: #fff; }
.hero p { margin: 0 0 14px; font-size: 1.05rem; line-height: 1.5; color: rgba(255,255,255,.94); max-width: 46rem; }
.hero .chips span { display: inline-block; margin: 0 8px 8px 0; padding: 4px 12px; border-radius: 999px;
        font-size: .82rem; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.32); }
.card { border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 18px 20px;
        background: var(--background-fill-secondary); }
.card .badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem;
        font-weight: 700; letter-spacing: .02em; color: #fff; background: var(--accent); }
.card .top { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-top: 10px; }
.card .top-label { font-size: 1.8rem; font-weight: 800; line-height: 1.15; }
.card .top-prob { font-size: 1.5rem; font-weight: 700; color: var(--accent); white-space: nowrap; }
.card .meter { height: 10px; margin: 12px 0 10px; border-radius: 999px; overflow: hidden;
        background: var(--border-color-primary); }
.card .meter span { display: block; height: 100%; border-radius: 999px; background: var(--accent); }
.card .hint { margin: 0 0 12px; color: var(--body-text-color-subdued); line-height: 1.45; }
.card .stats span { display: inline-block; margin: 0 8px 4px 0; padding: 3px 10px; border-radius: 8px;
        font-size: .8rem; border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.card .rows { margin: 4px 0 14px; }
.card .row { display: grid; grid-template-columns: minmax(90px, 34%) 1fr 56px; align-items: center; gap: 10px;
        padding: 3px 0; font-size: .92rem; color: var(--body-text-color-subdued); }
.card .row.first { color: var(--body-text-color); font-weight: 600; }
.card .row .bar { height: 8px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden; }
.card .row .bar i { display: block; height: 100%; border-radius: 999px; background: var(--border-color-primary-hover, #94a3b8); }
.card .row.first .bar i { background: var(--accent); }
.card .row .pct { text-align: right; font-variant-numeric: tabular-nums; }
.placeholder { text-align: center; padding: 40px 20px; color: var(--body-text-color-subdued); line-height: 1.6; }
.placeholder .big { font-size: 2.4rem; margin-bottom: 6px; }
.footer { text-align: center; padding: 14px 0 4px; font-size: .82rem; color: var(--body-text-color-subdued); }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def _to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert("RGB")


def _nice(label: str) -> str:
    return label[:1].upper() + label[1:]


def _card(pred) -> str:
    name, advice, colour = LEVELS[confidence_level(pred)]
    pct = pred.probs[0] * 100
    rows = "".join(
        f'<div class="row{" first" if i == 0 else ""}"><span class="name">{html.escape(_nice(lbl))}</span>'
        f'<span class="bar"><i style="width:{max(p * 100, 0.6):.1f}%"></i></span><span class="pct">{p * 100:.1f}%</span></div>'
        for i, (lbl, p) in enumerate(zip(pred.labels, pred.probs))
    )
    return f"""
<div class="card" style="--accent:{colour}">
  <span class="badge">{name}</span>
  <div class="top"><div class="top-label">{html.escape(_nice(pred.labels[0]))}</div><div class="top-prob">{pct:.1f}%</div></div>
  <div class="meter"><span style="width:{min(pct, 100):.1f}%"></span></div>
  <p class="hint">{html.escape(advice)}</p>
  <div class="rows">{rows}</div>
  <div class="stats"><span>⏱ {pred.latency_ms:.0f} ms</span><span>🎯 certainty {(1 - pred.entropy) * 100:.0f}%</span></div>
</div>"""


@gpu
def _infer(img: Image.Image):
    """The only GPU-bound step: top-5 prediction and the Grad-CAM map for the top answer."""
    pred = clf.predict(img, top_k=5)
    return pred, clf.grad_cam(img, pred.indices[0])


def classify(img: Image.Image | None, strength: float = DEFAULT_STRENGTH):
    if img is None:
        raise gr.Error("Please add an image first.")
    img = _to_rgb(img)
    pred, cam = _infer(img)
    crop = clf.model_view(img)
    heat = overlay_heatmap(crop, cam, alpha=strength)
    return _card(pred), crop, heat, (crop, cam)


def classify_url(url: str | None, strength: float = DEFAULT_STRENGTH):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste an image link (starting with http:// or https://) first.")
    try:
        img = load_image_from_url(url)  # fetched on CPU, outside the GPU slot
    except Exception as exc:
        raise gr.Error(f"Couldn't load an image from that link: {exc}") from exc
    return (*classify(img, strength), img)  # last output shows the loaded picture in the input box


def restyle(state, strength: float):
    """Re-draw the heatmap at a new strength without running the model again."""
    if state is None:
        return gr.skip()
    crop, cam = state
    return overlay_heatmap(crop, cam, alpha=strength)


def reset():
    return None, "", EMPTY_CARD, None, None, None


with gr.Blocks(title="Image Classifier") as demo:
    gr.HTML(HERO)
    state = gr.State()
    # Created now, placed in the right-hand column below, so the examples can sit under the upload box.
    out_card = gr.HTML(EMPTY_CARD, render=False)
    out_crop = gr.Image(label="Model's view", interactive=False, height=230, render=False)
    out_heat = gr.Image(label="Heatmap", interactive=False, height=230, render=False)
    outputs = [out_card, out_crop, out_heat, state]

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            inp = gr.Image(type="pil", label="Your image", sources=["upload", "webcam", "clipboard"], height=330)
            with gr.Row():
                url_in = gr.Textbox(
                    show_label=False, container=False, scale=4, placeholder="…or paste an image link"
                )
                url_btn = gr.Button("Load link", scale=1, min_width=110)
            with gr.Row():
                btn = gr.Button("Classify", variant="primary", scale=3, size="lg")
                clear = gr.Button("Clear", scale=1, size="lg")
            strength = gr.Slider(0.1, 1.0, value=DEFAULT_STRENGTH, step=0.05, label="Heatmap strength")
            gr.Examples(
                examples=[[os.path.join(HERE, "examples", f), DEFAULT_STRENGTH] for f, _ in EXAMPLES],
                example_labels=[label for _, label in EXAMPLES],
                inputs=[inp, strength],
                outputs=outputs,
                fn=classify,
                run_on_click=True,
                cache_examples=False,
                label="Try an example",
            )

        with gr.Column(scale=6, min_width=340):
            out_card.render()
            with gr.Row():
                out_crop.render()
                out_heat.render()

    with gr.Accordion("How to read the result", open=False):
        gr.Markdown(GUIDE)
    gr.HTML(FOOTER)

    btn.click(classify, [inp, strength], outputs)
    inp.upload(classify, [inp, strength], outputs)
    url_btn.click(classify_url, [url_in, strength], [*outputs, inp])
    url_in.submit(classify_url, [url_in, strength], [*outputs, inp])
    strength.release(restyle, [state, strength], out_heat)
    clear.click(reset, None, [inp, url_in, out_card, out_crop, out_heat, state])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
