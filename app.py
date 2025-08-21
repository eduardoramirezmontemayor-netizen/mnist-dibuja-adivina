import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="MNIST Dibuja y Adivina", layout="centered")

# Título y descripción
st.title("🖌️ MNIST: Dibuja y Adivina")
st.write("Dibuja un número del 0 al 9 y descubre qué número predice nuestro modelo.")

# Cargar modelo con caché
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_mnist.h5")

model = load_model()

# Crear lienzo interactivo
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

# Procesar imagen y predecir
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 0]):
    # Convertir a escala de grises y redimensionar
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert("L")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predicción
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Mostrar resultado
    st.subheader(f"🔢 El modelo predice: **{predicted_digit}**")
    st.image(img, caption="Imagen procesada", width=100)

    # Mostrar probabilidades en gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0], color="skyblue")
    ax.set_xticks(range(10))
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Probabilidad")
    ax.set_title("Confianza del modelo")
    st.pyplot(fig)
else:
    st.info("✏️ Dibuja algo en el lienzo para que el modelo pueda hacer una predicción.")
