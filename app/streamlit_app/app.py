"""
Brain MRI Tumor Analysis — Streamlit demo app.

A presentation-ready UI on top of the segmentation (ResNet34 U-Net3+) and
classification (ResNet34) pipeline trained on the BRISC 2025 dataset.

Run from the project root with:
    streamlit run app/streamlit_app/app.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# --- Make `src` importable regardless of the working directory Streamlit was launched from ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ClassificationModel, SegmentationModel  # noqa: E402
from src.utils import load_model, predict_mask, predict_tumor_class  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_tumor_bbox,
    draw_tumor_overlay,
    plot_tumor_report,
)

# ============================================================================
# Config
# ============================================================================
MODELS_DIR = PROJECT_ROOT / "models"
SEGMENTATION_WEIGHTS = MODELS_DIR / "best_model_seg.pth"
CLASSIFICATION_WEIGHTS = MODELS_DIR / "best_model_classif.pth"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Brain MRI Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Styling
# ============================================================================
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    #MainMenu, footer { visibility: hidden; }

    .app-header {
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 0.2rem;
    }
    .app-header h1 {
        font-size: 2.1rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1.4rem; }

    .status-pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 8px;
    }
    .pill-ok   { background: #dcfce7; color: #15803d; }
    .pill-bad  { background: #fee2e2; color: #b91c1c; }
    .pill-info { background: #dbeafe; color: #1d4ed8; }

    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .label { color: #64748b; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
    .metric-card .value { font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-top: 2px;}

    .class-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .class-name { width: 150px; font-size: 0.88rem; color: #334155; font-weight: 600; }
    .class-bar-bg { flex: 1; background: #eef2f7; border-radius: 8px; height: 12px; overflow: hidden; }
    .class-bar-fg { height: 100%; border-radius: 8px; }
    .class-pct { width: 52px; text-align: right; font-size: 0.85rem; color: #475569; font-weight: 600; }

    .verdict-tumor {
        background: linear-gradient(135deg, #fef2f2, #fff7ed);
        border: 1px solid #fecaca; border-radius: 16px; padding: 22px 26px;
    }
    .verdict-clean {
        background: linear-gradient(135deg, #f0fdf4, #ecfeff);
        border: 1px solid #bbf7d0; border-radius: 16px; padding: 22px 26px;
    }
    .verdict-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 4px; }
    .verdict-sub { color: #475569; font-size: 0.92rem; }

    .setup-box {
        background: #fffbeb; border: 1px solid #fde68a; border-radius: 14px;
        padding: 18px 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_classification_model():
    try:
        model = load_model(
            model_class=ClassificationModel,
            weights_path=CLASSIFICATION_WEIGHTS,
            model_kwargs={"num_classes": len(CLASS_NAMES), "in_channels": 1},
            device=DEVICE,
        )
        return model, None
    except Exception as e:
        return None, str(e)


def get_segmentation_model():
    try:
        model = load_model(
            model_class=SegmentationModel,
            weights_path=SEGMENTATION_WEIGHTS,
            model_kwargs={"in_channels": 1, "out_channels": 1},
            device=DEVICE,
        )
        return model, None
    except Exception as e:
        return None, str(e)


def run_full_pipeline(image: Image.Image) -> dict:
    """Runs classification, then segmentation (only if a tumor is predicted)."""
    clf_model, _ = get_classification_model()
    seg_model, _ = get_segmentation_model()

    t0 = time.perf_counter()
    classification = predict_tumor_class(
        model=clf_model, image_or_path=image, class_names=CLASS_NAMES, device=DEVICE
    )
    clf_time = time.perf_counter() - t0

    segmentation = None
    bbox_result = None
    overlay = None
    seg_time = 0.0

    if classification["has_tumor"] and seg_model is not None:
        t1 = time.perf_counter()
        segmentation = predict_mask(model=seg_model, image_or_path=image, device=DEVICE)

        mask = segmentation["predicted_mask"].numpy().astype(np.uint8)
        original_size = segmentation["original_size"]  # (W, H)
        mask_resized = np.array(
            Image.fromarray(mask * 255).resize(original_size, resample=Image.NEAREST)
        )
        mask_resized = (mask_resized > 0).astype(np.uint8)

        image_rgb = np.array(image.convert("RGB").resize(original_size))
        bbox_result = draw_tumor_bbox(image_rgb, mask_resized)
        overlay = draw_tumor_overlay(bbox_result["image_with_boxes"], mask_resized, alpha=0.35)
        segmentation["mask_resized"] = mask_resized
        segmentation["image_rgb"] = image_rgb
        seg_time = time.perf_counter() - t1

    return {
        "classification": classification,
        "segmentation": segmentation,
        "bbox_result": bbox_result,
        "overlay": overlay,
        "clf_time_ms": clf_time * 1000,
        "seg_time_ms": seg_time * 1000,
    }


def class_color(class_name: str) -> str:
    return {
        "glioma": "#ef4444",
        "meningioma": "#f97316",
        "pituitary": "#a855f7",
        "no_tumor": "#22c55e",
    }.get(class_name, "#2563eb")


def render_probability_bars(probabilities: dict[str, float]) -> None:
    for name, pct in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        color = class_color(name)
        st.markdown(
            f"""
            <div class="class-row">
                <div class="class-name">{CLASS_LABELS.get(name, name)}</div>
                <div class="class-bar-bg">
                    <div class="class-bar-fg" style="width:{pct}%; background:{color};"></div>
                </div>
                <div class="class-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### 🧠 Brain MRI Analysis")
    st.caption("Segmentation + Classification pipeline · BRISC 2025")

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. **Classify** the scan (glioma / meningioma / pituitary / no tumor)\n"
        "2. If a tumor is detected, **segment** it (ResNet34 U-Net3+)\n"
        "3. Localize regions, draw bounding boxes & overlay"
    )

    st.markdown("---")
    st.markdown("**System status**")

    clf_model, clf_err = get_classification_model()
    seg_model, seg_err = get_segmentation_model()

    device_pill = "pill-ok" if DEVICE == "cuda" else "pill-info"
    st.markdown(
        f'<span class="status-pill {device_pill}">Device: {DEVICE.upper()}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-pill {"pill-ok" if clf_model else "pill-bad"}">'
        f'Classifier: {"Ready" if clf_model else "Not loaded"}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-pill {"pill-ok" if seg_model else "pill-bad"}">'
        f'Segmentation: {"Ready" if seg_model else "Not loaded"}</span>',
        unsafe_allow_html=True,
    )

    with st.expander("Architecture details"):
        st.markdown(
            "- **Classifier:** ResNet34 backbone, single-channel input\n"
            "- **Segmentation:** ResNet34 encoder + UNet3+ full-scale skip "
            "connections decoder\n"
            "- **Classes:** glioma, meningioma, pituitary, no_tumor"
        )

    st.markdown("---")
    st.caption("Built by Hossein Heydari · github.com/HosseinHeydari2004")

# ============================================================================
# Header
# ============================================================================
st.markdown(
    '<div class="app-header"><h1>🧠 Brain MRI Tumor Analysis</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">AI-assisted tumor classification and segmentation '
    'for brain MRI scans — for research &amp; demonstration purposes only, '
    'not a medical diagnostic device.</div>',
    unsafe_allow_html=True,
)

if clf_err or seg_err:
    st.markdown(
        f"""
        <div class="setup-box">
        <b>⚠️ Model weights not found.</b><br/>
        Place your trained checkpoints in <code>models/</code> as described in the README, then reload this page:
        <ul>
            <li><code>models/best_model_classif.pth</code> {"✅" if clf_model else "❌ missing"}</li>
            <li><code>models/best_model_seg.pth</code> {"✅" if seg_model else "❌ missing"}</li>
        </ul>
        You can still browse the interface below, but analysis is disabled until the weights are in place.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

# ============================================================================
# Upload + analyze
# ============================================================================
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("#### 1 · Upload an MRI scan")
    uploaded_file = st.file_uploader(
        "Drag and drop a brain MRI slice (PNG / JPG)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(image, caption="Uploaded scan", use_container_width=True)

    analyze_clicked = st.button(
        "🔍 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=(image is None) or (clf_model is None),
    )

with right:
    st.markdown("#### 2 · Results")

    if "result" not in st.session_state:
        st.session_state["result"] = None

    if analyze_clicked and image is not None:
        with st.spinner("Running inference..."):
            st.session_state["result"] = run_full_pipeline(image)

    result = st.session_state["result"]

    if result is None:
        st.info("Upload a scan and click **Run Analysis** to see results here.")
    else:
        clf = result["classification"]
        seg = result["segmentation"]
        has_tumor = clf["has_tumor"]

        verdict_class = "verdict-tumor" if has_tumor else "verdict-clean"
        verdict_icon = "🔴" if has_tumor else "🟢"
        verdict_text = (
            f"{CLASS_LABELS.get(clf['predicted_class'], clf['predicted_class'])} detected"
            if has_tumor
            else "No tumor detected"
        )
        st.markdown(
            f"""
            <div class="{verdict_class}">
                <div class="verdict-title">{verdict_icon} {verdict_text}</div>
                <div class="verdict-sub">Classification confidence: {clf['confidence']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**Class probabilities**")
        render_probability_bars(clf["probabilities"])

        st.markdown("")
        cols = st.columns(4)
        cols[0].markdown(
            metric_card("Predicted Class", CLASS_LABELS.get(clf["predicted_class"], clf["predicted_class"])),
            unsafe_allow_html=True)
        cols[1].markdown(metric_card("Confidence", f"{clf['confidence']:.1f}%"), unsafe_allow_html=True)
        if seg is not None:
            cols[2].markdown(metric_card("Tumor Coverage", f"{seg['tumor_coverage']:.2f}%"), unsafe_allow_html=True)
            cols[3].markdown(metric_card("Regions Found", str(result["bbox_result"]["num_regions"])),
                             unsafe_allow_html=True)
        else:
            cols[2].markdown(metric_card("Tumor Coverage", "—"), unsafe_allow_html=True)
            cols[3].markdown(metric_card("Regions Found", "—"), unsafe_allow_html=True)

# ============================================================================
# Detailed segmentation report (full width, below)
# ============================================================================
result = st.session_state.get("result")
if result is not None and result["segmentation"] is not None:
    st.markdown("---")
    st.markdown("#### 3 · Segmentation report")

    seg = result["segmentation"]
    tab1, tab2 = st.tabs(["Visual report", "Region details"])

    with tab1:
        fig = plot_tumor_report(
            image_rgb=seg["image_rgb"],
            mask=seg["mask_resized"],
            bbox_result=result["bbox_result"],
        )
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Download report (PNG)",
            data=buf.getvalue(),
            file_name="brain_mri_analysis_report.png",
            mime="image/png",
        )

    with tab2:
        boxes = result["bbox_result"]["boxes"]
        if not boxes:
            st.write("No distinct regions above the minimum area threshold.")
        else:
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                st.markdown(f"**Region #{i}** — position `(x={x}, y={y})`, size `{w}×{h}px`")

    st.caption(
        f"Inference time — classification: {result['clf_time_ms']:.0f} ms · "
        f"segmentation: {result['seg_time_ms']:.0f} ms · device: {DEVICE.upper()}"
    )

st.markdown("---")
st.caption(
    "⚠️ This tool is a research/portfolio demonstration and is **not** intended for clinical use. "
    "All predictions must be reviewed by a qualified medical professional."
)
