"""Vision Lab — multi-model ImageNet classifier with Grad-CAM explanations (Streamlit)."""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from classifier import (
    COLORMAP_NAMES,
    DEFAULT_MODEL,
    MODEL_SPECS,
    UNRELIABLE_BELOW,
    ImageClassifier,
    confidence_level,
    heatmap_only,
    load_image_from_bytes,
    load_image_from_url,
    overlay_heatmap,
)

st.set_page_config(page_title="Vision Lab · Explainable image classifier", page_icon="🔬", layout="wide")

ACCENT = "#7C5CFF"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1250px; }}
      .hero {{
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #4f2fd8 0%, {ACCENT} 45%, #14b8a6 100%);
        color: white; box-shadow: 0 8px 30px rgba(124,92,255,.28);
      }}
      .hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; color: white; }}
      .hero p  {{ margin: .35rem 0 0; opacity: .92; font-size: 1.02rem; }}
      .chip {{ display:inline-block; padding: .15rem .65rem; margin: .5rem .35rem 0 0; border-radius: 999px;
               background: rgba(255,255,255,.18); font-size: .8rem; backdrop-filter: blur(4px); }}
      .top1 {{ padding: 1.1rem 1.3rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25);
               background: linear-gradient(135deg, rgba(124,92,255,.12), rgba(20,184,166,.10)); }}
      .top1 .lbl {{ font-size: 1.9rem; font-weight: 800; line-height: 1.15; text-transform: capitalize; }}
      .top1 .sub {{ opacity: .75; font-size: .9rem; margin-top: .25rem; }}
      .bar {{ height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); margin-top: .7rem; overflow: hidden; }}
      .bar > div {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, {ACCENT}, #14b8a6); }}
      .badge {{ display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; }}
      .good {{ background: rgba(34,197,94,.18); color:#16a34a; }}
      .mid  {{ background: rgba(245,158,11,.20); color:#d97706; }}
      .low  {{ background: rgba(239,68,68,.18); color:#dc2626; }}
      .warn {{ margin-top:.7rem; padding:.6rem .9rem; border-radius:12px; font-size:.88rem; line-height:1.35;
               background: rgba(245,158,11,.14); border:1px solid rgba(245,158,11,.35); }}
      div[data-testid="stMetric"] {{ background: rgba(128,128,128,.08); border-radius: 12px; padding: .6rem .9rem; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
      div[data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; }}
      div[data-testid="stMetricLabel"] p {{ font-size: .78rem; }}
      .cap {{ text-align:center; font-size:.85rem; opacity:.7; margin-top:-.4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def _load_model(model_key: str) -> ImageClassifier:
    clf = ImageClassifier(model_key)
    clf.predict(Image.new("RGB", (clf.input_size, clf.input_size)), top_k=1)  # warm-up so the first real run is fast
    return clf


def get_model(model_key: str) -> ImageClassifier:
    spec = MODEL_SPECS[model_key]
    with st.spinner(f"Loading {spec.name} (first use downloads ~{spec.size_mb:.0f} MB of weights)…"):
        return _load_model(model_key)


def digest(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


@st.cache_data(show_spinner=False, max_entries=128)
def cached_predict(_img: Image.Image, key: str, model_key: str, top_k: int, tta: bool):
    return _load_model(model_key).predict(_img, top_k=top_k, tta=tta)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_cam(_img: Image.Image, key: str, model_key: str, class_idx: int):
    return _load_model(model_key).grad_cam(_img, class_idx)


def model_label(k: str) -> str:
    s = MODEL_SPECS[k]
    return f"{s.name} · {s.acc1:.1f}% top-1"


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_key = st.selectbox(
        "Model", list(MODEL_SPECS), index=list(MODEL_SPECS).index(DEFAULT_MODEL), format_func=model_label,
        help="All models are pretrained on ImageNet-1k. Bigger ones are more accurate but slower, and the first use of each downloads its weights.",
    )
    clf = get_model(model_key)
    spec = clf.spec
    top_k = st.slider("Top-K predictions", 3, 10, 5)
    tta = st.toggle("Test-time augmentation", value=False, help="Average predictions over the image and its mirror image. Slightly slower, slightly more robust.")
    st.divider()
    st.subheader("🔥 Grad-CAM")
    show_cam = st.toggle("Explain predictions", value=True, help="Highlights the pixels that pushed the network towards a class.")
    cmap = st.selectbox("Colormap", COLORMAP_NAMES, index=0, disabled=not show_cam)
    alpha = st.slider("Overlay strength", 0.1, 1.0, 0.55, 0.05, disabled=not show_cam)
    st.divider()
    with st.expander("ℹ️ About the model"):
        st.markdown(
            f"""
            **{spec.name}** · torchvision `{spec.weights.name}`
            - Parameters: **{clf.n_params/1e6:.1f} M**
            - ImageNet-1k top-1 accuracy: **{spec.acc1:.1f}%**
            - Classes: **1,000** (ImageNet-1k)
            - Input: {clf.input_size}×{clf.input_size} (resize → centre-crop)
            - Device: **{str(clf.device).upper()}**

            It can only name the 1,000 ImageNet categories, so unusual subjects get the *closest* match.
            When the top probability is under {UNRELIABLE_BELOW:.0%} the app flags the result as unreliable.
            """
        )
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        st.subheader("🕘 Recent")
        for h in reversed(st.session_state.history[-8:]):
            st.caption(f"**{h['label']}** · {h['conf']:.0%} — {h['name']}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ------------------------------------------------------------------------------ hero
st.markdown(
    f"""
    <div class="hero">
      <h1>🔬 Vision Lab</h1>
      <p>Drop in any photo the model tells you what it sees, <b>how sure it is</b>, and <b>where it looked</b>.</p>
      <span class="chip">PyTorch</span><span class="chip">{spec.name}</span><span class="chip">Grad-CAM</span>
      <span class="chip">Streamlit</span><span class="chip">{str(clf.device).upper()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- input
items: list[tuple[str, Image.Image]] = []
tab_up, tab_url, tab_cam = st.tabs(["📁 Upload", "🔗 From URL", "📷 Camera"])

with tab_up:
    files = st.file_uploader(
        "Upload one or more images", type=["jpg", "jpeg", "png", "webp", "bmp", "gif"], accept_multiple_files=True
    )
    for f in files or []:
        try:
            items.append((f.name, load_image_from_bytes(f.getvalue())))
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read **{f.name}**: {e}")

with tab_url:
    # `?url=https://…` pre-fills this box, so a result can be shared or bookmarked as a link.
    url = st.text_input("Image URL", value=st.query_params.get("url", ""), placeholder="https://example.com/photo.jpg")
    if url.strip():
        try:
            with st.spinner("Downloading…"):
                items.append((url.strip().rsplit("/", 1)[-1][:40] or "url-image", load_image_from_url(url.strip())))
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't fetch that image: {e}")

with tab_cam:
    # Gated behind a toggle so the browser doesn't ask for camera access on every page load.
    if st.toggle("Enable camera", value=False):
        snap = st.camera_input("Take a picture")
        if snap is not None:
            items.append(("camera.jpg", load_image_from_bytes(snap.getvalue())))

if not items:
    st.info("👆 Add an image above to get started. Tip: photos with one clear subject work best.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Categories", "1,000")
    c2.metric("Parameters", f"{clf.n_params/1e6:.1f} M")
    c3.metric("ImageNet top-1", f"{spec.acc1:.1f}%")
    st.stop()


# --------------------------------------------------------------------------- results
BADGES = {
    "high": '<span class="badge good">High confidence</span>',
    "moderate": '<span class="badge mid">Moderate confidence</span>',
    "uncertain": '<span class="badge low">Uncertain — top guesses are close</span>',
    "unreliable": '<span class="badge low">Unreliable — likely outside the 1,000 classes</span>',
}
UNRELIABLE_NOTE = (
    f"⚠️ <b>Low confidence (&lt;{UNRELIABLE_BELOW:.0%}).</b> This image probably isn't one of the 1,000 ImageNet "
    "categories, so these labels are the <i>nearest</i> matches, not answers. Check the Grad-CAM to see what the model latched onto."
)


def render_comparison(img: Image.Image, key: str, uid: str) -> None:
    """Run the same image through other models and tabulate the results."""
    with st.expander("⚖️ Compare models on this image"):
        others = [k for k in MODEL_SPECS if k != model_key]
        picks = st.multiselect(
            "Add models", others, key=f"cmp-{uid}", format_func=model_label,
            help="Each model is downloaded the first time you select it.",
        )
        if not picks:
            st.caption("Pick one or more models to see how they disagree.")
            return
        rows = []
        for k in [model_key, *picks]:
            get_model(k)
            p = cached_predict(img, key, k, 3, tta)
            s = MODEL_SPECS[k]
            rows.append(
                {
                    "Model": s.name + (" (current)" if k == model_key else ""),
                    "Top-1": p.labels[0].title(),
                    "Confidence": p.probs[0],
                    "Runner-up": p.labels[1].title(),
                    "Latency (ms)": round(p.latency_ms),
                    "Params (M)": round(s.params_m, 1),
                    "ImageNet top-1 (%)": round(s.acc1, 1),
                }
            )
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
        )


def render_one(name: str, img: Image.Image, uid: str) -> dict:
    key = digest(img)
    pred = cached_predict(img, key, model_key, top_k, tta)
    view = clf.model_view(img)
    top_p = pred.probs[0]
    level = confidence_level(pred)
    warn_html = f'<div class="warn">{UNRELIABLE_NOTE}</div>' if level == "unreliable" else ""

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="top1">
              <div class="sub">Top prediction</div>
              <div class="lbl">{pred.labels[0]}</div>
              <div class="bar"><div style="width:{top_p*100:.1f}%"></div></div>
              <div class="sub">{top_p:.1%} &nbsp;·&nbsp; {BADGES[level]}</div>{warn_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{pred.latency_ms:.0f} ms")
        m2.metric("Entropy", f"{pred.entropy:.2f}", help="Normalised entropy of the full 1,000-way distribution (0 = certain, 1 = uniform).")
        m3.metric("Input size", f"{img.width}×{img.height}")

        fig = go.Figure(
            go.Bar(
                x=[p * 100 for p in pred.probs][::-1],
                y=[l.title() for l in pred.labels][::-1],
                orientation="h",
                marker=dict(color=[p for p in pred.probs][::-1], colorscale=[[0, "#14b8a6"], [1, ACCENT]]),
                text=[f"{p:.1%}" for p in pred.probs][::-1],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            height=60 + 38 * len(pred.labels),
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(title="Probability (%)", range=[0, max(pred.probs) * 100 * 1.18], gridcolor="rgba(128,128,128,.2)"),
            yaxis=dict(automargin=True),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", key=f"chart-{uid}", config={"displayModeBar": False})

    with right:
        if show_cam:
            target_label = st.selectbox(
                "Explain class", [f"{l.title()} ({p:.1%})" for l, p in zip(pred.labels, pred.probs)], key=f"cls-{uid}",
                help="Pick which prediction Grad-CAM should explain.",
            )
            cls_pos = [f"{l.title()} ({p:.1%})" for l, p in zip(pred.labels, pred.probs)].index(target_label)
            with st.spinner("Computing Grad-CAM…"):
                cam = cached_cam(img, key, model_key, pred.indices[cls_pos])
            view_mode = st.radio("View", ["Overlay", "Side by side", "Heatmap"], horizontal=True, key=f"vm-{uid}", label_visibility="collapsed")
            ov = overlay_heatmap(view, cam, alpha, cmap)
            if view_mode == "Overlay":
                st.image(ov.resize((448, 448)), width="stretch")
                st.markdown('<div class="cap">Warm regions influenced this class the most</div>', unsafe_allow_html=True)
            elif view_mode == "Heatmap":
                st.image(heatmap_only(cam, (448, 448), cmap), width="stretch")
            else:
                a, b = st.columns(2)
                a.image(view.resize((448, 448)), caption=f"What the model sees ({clf.input_size}×{clf.input_size} crop)", width="stretch")
                b.image(ov.resize((448, 448)), caption="Grad-CAM", width="stretch")
        else:
            st.image(view.resize((448, 448)), caption=f"What the model sees ({clf.input_size}×{clf.input_size} crop)", width="stretch")

    render_comparison(img, key, uid)
    return {"name": name, "label": pred.labels[0].title(), "conf": top_p, "pred": pred}


results = []
if len(items) == 1:
    results.append(render_one(*items[0], uid="0"))
else:
    st.success(f"Analysing **{len(items)}** images")
    tabs = st.tabs([f"{i+1}. {n[:18]}" for i, (n, _) in enumerate(items)])
    for i, (tab, (n, im)) in enumerate(zip(tabs, items)):
        with tab:
            results.append(render_one(n, im, uid=str(i)))

    st.subheader("📊 Batch summary")
    df = pd.DataFrame(
        [
            {
                "Image": r["name"],
                "Prediction": r["label"],
                "Confidence": r["conf"],
                "Runner-up": r["pred"].labels[1].title(),
                "Runner-up conf.": r["pred"].probs[1],
            }
            for r in results
        ]
    )
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Runner-up conf.": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")

# record history (once per unique batch)
batch_key = "|".join(f"{n}:{digest(i)[:8]}" for n, i in items)
if st.session_state.get("last_batch") != batch_key:
    st.session_state.last_batch = batch_key
    for r in results:
        st.session_state.history.append({"name": r["name"], "label": r["label"], "conf": r["conf"]})
    st.rerun()
