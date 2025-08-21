import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("modelo_mnist.h5")
#model = keras.models.load_model("modelo_mnist.keras")

st.set_page_config(page_title="Reconocimiento de Dígitos", page_icon="🧠")
st.title("🧠 Reconocimiento de Dígitos Dibujados")
st.markdown("Dibuja un número del 0 al 9 en el lienzo y deja que el modelo lo adivine.")

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

# Función para centrar el dígito en la imagen
def centrar_imagen(imagen):
    coords = np.column_stack(np.where(imagen > 0.1))
    if coords.size == 0:
        return imagen
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    recorte = imagen[y_min:y_max+1, x_min:x_max+1]
    nueva = np.zeros((28, 28))
    h, w = recorte.shape
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    nueva[y_offset:y_offset+h, x_offset:x_offset+w] = recorte
    return nueva

# Procesar la imagen si hay dibujo
if canvas_result.image_data is not None:
    # Convertir a imagen PIL en escala de grises
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

    # Redimensionar a 28x28
    img = img.resize((28, 28), Image.ANTIALIAS)

    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0

    # Centrar el dígito
    img_array = centrar_imagen(img_array)

    # Mostrar imagen procesada
    st.image(img_array, caption="🖼 Imagen procesada", width=100)

    # Preparar para predicción
    input_data = img_array.reshape(1, 28, 28, 1)

    # Predecir
    prediction = model.predict(input_data)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    # Mostrar resultado
    st.subheader(f"🔢 Predicción: {predicted_digit}")
    st.caption(f"Confianza del modelo: {confidence:.2f}%")

    # Mostrar las 3 clases más probables
    top_3 = np.argsort(prediction[0])[-3:][::-1]
    st.markdown("### 📊 Top 3 predicciones:")
    for i in top_3:
        st.write(f"- {i}: {prediction[0][i]*100:.2f}%")

else:
    st.info("🖌 Dibuja un número en el lienzo para comenzar.")
