import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT_DIR))
    from predictor import load_model, predict_with_gradcam
    from utils import get_asset_path
else:
    from app.predictor import load_model, predict_with_gradcam
    from app.utils import get_asset_path

from src.training.evaluate import (
    build_evaluation_generator,
    evaluate_generator,
    plot_confusion_matrix,
    plot_confidence_distribution,
)


def parse_prediction_label(label: str) -> tuple[str, str, bool]:
    if ' — ' in label:
        plant, disease = label.split(' — ', 1)
    else:
        plant, disease = label, ''
    is_healthy = disease.lower() == 'healthy'
    return plant, disease, is_healthy


def render_prediction_status(confidence: float, is_healthy: bool) -> None:
    if confidence >= 0.80:
        st.success('✅ Reliable prediction')
    elif confidence >= 0.50:
        st.warning('⚠️ Low confidence prediction. Please upload a clear image of the affected leaf.')
    else:
        st.error('❌ Unable to confidently classify this image. Please upload a clear leaf image and try again.')


@st.cache_resource
def get_cached_model():
    try:
        return load_model()
    except FileNotFoundError:
        st.warning('No trained model file was found in the models directory. Train a model or place a .keras model there before predicting.')
        return None

st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition', 'Evaluation'])

logo_path = get_asset_path('Disease.png')
if logo_path:
    st.image(str(logo_path))
else:
    st.warning('Logo image not found in assets/Disease.png.')

if app_mode == 'Home':
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Prediction System for Sustainable Agriculture')
    uploaded_file = st.file_uploader('Choose an Image:')

    uploaded_image = None
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert('RGB')
        st.image(uploaded_image, width=400, use_container_width=True)

    if st.button('Predict'):
        if uploaded_image is None:
            st.warning('Please upload an image before predicting.')
        else:
            st.write('### Prediction')
            try:
                model = get_cached_model()
                if model is None:
                    st.stop()
                _, top_confidence, top_predictions, gradcam_overlay = predict_with_gradcam(uploaded_image, model=model)
                top_label, top_confidence = top_predictions[0]
                plant_name, disease_name, is_healthy = parse_prediction_label(top_label)

                st.markdown(f"**{plant_name}**")
                if is_healthy:
                    st.markdown('🟢 **Healthy**')
                    st.markdown(f"**{disease_name}**")
                else:
                    st.markdown('🔴 **Disease detected**')
                    st.markdown(f"**{disease_name}**")

                st.write('Confidence:')
                st.progress(int(top_confidence * 100))
                st.markdown(f"**{top_confidence * 100:.1f}%**")
                render_prediction_status(top_confidence, is_healthy)

                if top_confidence < 0.50:
                    st.write('The model is not confident enough to make a reliable prediction. Please try a clearer leaf image.')

                st.write('### Top predictions')
                top_table = "| Disease | Confidence |\n|---|---|\n"
                for label, score in top_predictions:
                    top_table += f"| {label} | {score * 100:.1f}% |\n"
                st.markdown(top_table)

                if not is_healthy and top_confidence >= 0.50:
                    st.write('### General guidance')
                    st.write('- Isolate severely affected plants')
                    st.write('- Monitor surrounding plants')
                    st.write('- Consult an agricultural professional for treatment decisions')

                st.write('### Why the model predicted this')
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_image, caption='Original image', use_column_width=True)
                with col2:
                    st.image(gradcam_overlay, caption='Grad-CAM overlay', use_column_width=True)
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')

elif app_mode == 'Evaluation':
    st.header('Model Evaluation')
    data_dir = st.text_input('Validation dataset directory (Organized by class subfolders)')
    batch_size = st.number_input('Batch size', value=32, min_value=1, max_value=128, step=1)
    top_n = st.number_input('Top misclassified examples', value=10, min_value=1, max_value=50, step=1)

    if st.button('Run Evaluation'):
        if not data_dir:
            st.warning('Please provide the validation dataset directory.')
        else:
            try:
                model = get_cached_model()
                generator = build_evaluation_generator(data_dir, batch_size=batch_size)
                stats = evaluate_generator(model, generator, top_n_misclassified=top_n)

                st.write('### Classification metrics')
                st.write('**Accuracy:**', f"{stats['accuracy'] * 100:.2f}%")
                st.write('**Precision (macro):**', f"{stats['precision_macro'] * 100:.2f}%")
                st.write('**Recall (macro):**', f"{stats['recall_macro'] * 100:.2f}%")
                st.write('**F1-score (macro):**', f"{stats['f1_macro'] * 100:.2f}%")
                st.write('**F1-score (weighted):**', f"{stats['f1_weighted'] * 100:.2f}%")
                if stats.get('roc_auc_macro') is not None:
                    st.write('**ROC AUC (macro):**', f"{stats['roc_auc_macro'] * 100:.2f}%")
                    st.write('**ROC AUC (weighted):**', f"{stats['roc_auc_weighted'] * 100:.2f}%")

                st.write('### Per-class accuracy')
                for label, value in stats['per_class_accuracy'].items():
                    st.write(f"- {label}: {value * 100:.2f}%")

                st.write('### Confusion Matrix')
                st.pyplot(plot_confusion_matrix(stats['confusion_matrix'], stats['labels']))

                st.write('### Confidence distribution')
                counts, edges = stats['confidence_distribution']
                st.pyplot(plot_confidence_distribution(counts, edges))

                st.write('### Misclassified examples')
                for example in stats['misclassified_examples']:
                    item_text = f"{example['true_label']} → {example['predicted_label']} ({example['confidence'] * 100:.1f}%)"
                    if 'filepath' in example:
                        item_text += f" — {example['filepath']}"
                    st.write(item_text)
            except Exception as exc:
                st.error(f'Evaluation failed: {exc}')
