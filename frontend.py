import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "model2.keras"

CLASS_NAMES = [
    "Corona Virus Disease",
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]

IMG_SIZE = (224, 224)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Lung X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f5f5;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


MODEL = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🫁 Lung X-Ray Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a chest X-ray for 4-class classification</div>',
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    # Resize
    image_resized = image.resize(IMG_SIZE)

    # Convert to numpy
    image_array = np.array(image_resized, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # DenseNet preprocessing
    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    # Prediction
    predictions = MODEL.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index] * 100

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">
                {predicted_class}
            </div>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # ALL CLASS PROBABILITIES
    # =====================================================

    st.subheader("Prediction Probabilities")

    for class_name, probability in zip(CLASS_NAMES, predictions):

        st.write(
            f"**{class_name}** — {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Deep learning model: DenseNet121 | Input size: 224×224"
)