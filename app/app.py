import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predictor import (
    load_model,
    predict_with_confidence,
)


MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "mobilenet_benchmark_final.keras"
)

TOP_K = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():
    """Load and cache the final trained model."""

    return load_model(MODEL_PATH)


# ============================================================
# LABEL HELPERS
# ============================================================

def parse_prediction_label(label: str):
    """Split readable label into plant and disease."""

    separator = " — "

    if separator in label:
        plant, disease = label.split(
            separator,
            1,
        )
    else:
        plant = label
        disease = ""

    return (
        plant.strip(),
        disease.strip(),
        disease.strip().lower() == "healthy",
    )


def render_confidence_status(confidence: float):
    """Display confidence level."""

    if confidence >= 0.90:
        st.success(
            f"High confidence: {confidence * 100:.2f}%"
        )

    elif confidence >= 0.70:
        st.warning(
            f"Moderate confidence: {confidence * 100:.2f}%"
        )

    else:
        st.error(
            f"Low confidence: {confidence * 100:.2f}%"
        )


# ============================================================
# APP
# ============================================================

def main():

    st.title("🌿 Plant Disease Classification")

    st.write(
        "Upload a plant leaf image to identify the "
        "most likely disease."
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("Model Information")

    st.sidebar.write(
        "Architecture: MobileNetV3Small"
    )

    st.sidebar.write(
        "Classes: 38"
    )

    st.sidebar.write(
        "Input: 224 × 224"
    )

    st.sidebar.write(
        "Parameters: 961,046"
    )

    st.sidebar.write(
        "Full validation accuracy: 89.78%"
    )

    # --------------------------------------------------------
    # MODEL LOAD
    # --------------------------------------------------------

    try:
        model = get_model()

    except Exception as exc:
        st.error("Unable to load the trained model.")
        st.exception(exc)
        return

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to get a prediction."
        )
        return

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as exc:
        st.error("Unable to read the uploaded image.")
        st.exception(exc)
        return

    left, right = st.columns(2)

    with left:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing image..."):

        try:
            (
                predicted_index,
                confidence,
                predictions,
                status,
            ) = predict_with_confidence(
                image,
                model=model,
                top_k=TOP_K,
            )

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)
            return

    # --------------------------------------------------------
    # PRIMARY RESULT
    # --------------------------------------------------------

    with right:

        st.subheader("Prediction")

        primary = predictions[0]

        label = primary["label"]

        plant, disease, is_healthy = (
            parse_prediction_label(label)
        )

        st.markdown(
            f"### {label}"
        )

        render_confidence_status(
            confidence
        )

        if is_healthy:

            st.success(
                f"The model predicts that the "
                f"{plant} leaf is healthy."
            )

        else:

            st.warning(
                f"Plant: **{plant}**"
            )

            st.error(
                f"Detected condition: **{disease}**"
            )

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    st.divider()

    st.subheader("Top-5 Predictions")

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):

        label = prediction["label"]

        score = prediction["confidence"]

        st.write(
            f"**{rank}. {label}** — "
            f"{score * 100:.2f}%"
        )

        st.progress(
            min(
                max(score, 0.0),
                1.0,
            )
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "This AI prediction is informational and should "
        "not replace professional agricultural diagnosis."
    )


if __name__ == "__main__":
    main()