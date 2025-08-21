
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Cargar modelo previamente subido
model = tf.keras.models.load_model("modelo_mnist.h5")

st.set_page_config(page_title="MNIST Deba", layout="centered")
st.title("🖌️ MNIST: Dibuja y Adivina")
st.write("Dibuja un número del 0 al 9 y descubre qué número predice nuestro modelo.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert("L")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    st.subheader(f"🔢 Predicción: {np.argmax(prediction)}")
    st.image(img, caption="Imagen procesada", width=100)
