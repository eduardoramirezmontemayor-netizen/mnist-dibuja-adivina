import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("modelo_mnist.h5")

st.title("🧠 Reconocimiento de Dígitos Dibujados")

# Crear el lienzo para dibujar
canvas_result = st_canvas(
    fill_color="rgb(0,0,0)",            # Color de relleno (negro)
    stroke_width=10,                    # Grosor del trazo
    stroke_color="rgb(255,255,255)",    # Color del trazo (blanco)
    background_color="rgb(0,0,0)",      # Fondo del lienzo (negro)
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Procesar la imagen si hay dibujo
if canvas_result.image_data is not None:
    # Convertir a imagen PIL en escala de grises
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

    # Invertir colores para que el fondo sea negro y el número blanco
    img = ImageOps.invert(img)

    # Redimensionar a 28x28 píxeles
    img = img.resize((28, 28))

    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0

    # Aplicar umbral para mejorar contraste
    img_array[img_array < 0.5] = 0
    img_array[img_array >= 0.5] = 1

    # Mostrar imagen procesada
    st.image(img_array, caption="Imagen procesada", width=100)

    # Preparar para predicción
    input_data = img_array.reshape(1, 28, 28, 1)

    # Predecir
    prediction = model.predict(input_data)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    # Mostrar resultado
    st.subheader(f"🔢 Predicción: {predicted_digit}")
    st.caption(f"Confianza del modelo: {confidence:.2f}%")
