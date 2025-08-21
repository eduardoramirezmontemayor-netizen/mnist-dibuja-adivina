import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("modelo_mnist.h5")

st.title("Reconocimiento de Dígitos 🧠✍️")

canvas_result = st.canvas(
    fill_color="rgb(0,0,0)",  # Color del fondo del lienzo
    stroke_width=10,
    stroke_color="rgb(255,255,255)",  # Color del trazo (blanco sobre negro)
    background_color="rgb(0,0,0)",  # Fondo negro
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convertir imagen a escala de grises
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

    # Invertir colores para que el fondo sea negro y el trazo blanco
    img = ImageOps.invert(img)

    # Redimensionar a 28x28 píxeles
    img = img.resize((28, 28))

    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0

    # Aplic
