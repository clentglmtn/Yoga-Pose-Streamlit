import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Yoga Pose Estimation",
    layout="centered"
)

st.title("Yoga Pose Estimation using YOLOv26")
st.write(
    "Upload a Yoga Pose Image"
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model.predict(temp_file.name, conf=0.25)

    result_image = results[0].plot()

    st.subheader("Prediction Result")
    st.image(result_image, use_container_width=True)