import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Reconstruir el modelo manualmente
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights("modelo_mnist.weights.h5")

st.set_page_config(page_title="Reconocimiento de Dígitos", page_icon="🧠")
st.title("🧠 Reconocimiento de Dígitos Dibujados")
st.markdown("Dibuja un número del 0 al 9 en el lienzo y deja que el modelo lo adivine.")

# Crear el lienzo para dibujar
canvas_result = st_canvas(
    fill_color="rgb(0,0,0)",
    stroke_width=10,
    stroke_color="rgb(255,255,255)",
    background_color="rgb(0,0,0)",
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
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.array(img) / 255.0
    img_array = centrar_imagen(img_array)
    st.image(img_array, caption="🖼 Imagen procesada", width=100)
    input_data = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(input_data)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.subheader("🔍 Resultado del modelo:")
    if confidence > 90:
        st.success(f"🎯 ¡Acierto total! El modelo cree que has dibujado un **{predicted_digit}** con {confidence:.2f}% de confianza.")
        st.balloons()
    elif confidence > 70:
        st.info(f"🤔 El modelo cree que es un **{predicted_digit}**, pero no está 100% seguro ({confidence:.2f}%).")
    else:
        st.warning(f"😅 El modelo está confundido... cree que es un **{predicted_digit}**, pero solo con {confidence:.2f}% de confianza.")

    # Mostrar las 3 clases más probables
    top_3 = np.argsort(prediction[0])[-3:][::-1]
    st.markdown("### 📊 Top 3 predicciones:")
    for i in top_3:
        st.write(f"- {i}: {prediction[0][i]*100:.2f}%")
else:
    st.info("🖌 Dibuja un número en el lienzo para comenzar.")
