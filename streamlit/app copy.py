import streamlit as st
import cv2
import time
from model_utils import get_model, detectar_personas

st.set_page_config(layout="centered", page_title="Detector de Personas - Biblioteca UIDE")

st.title("🎥 Detector de Personas en la Biblioteca - UIDE")
st.write("Esta aplicación detecta y cuenta personas en tiempo real usando modelos de visión artificial.")

# Inicializar el estado si no existe
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Selección del modelo
st.subheader("🧠 Modo de detección")
modelo_seleccionado = st.radio(
    "Seleccione el modelo:",
    ["Faster R-CNN (ResNet50)", "SSD MobileNetV2"],
    disabled=st.session_state.camera_active  # Deshabilitar mientras la cámara esté activa
)

modelo = get_model(modelo_seleccionado)

# Botones de control
col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Activar cámara", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
with col2:
    if st.button("🛑 Detener cámara", disabled=not st.session_state.camera_active):
        st.session_state.camera_active = False

video_placeholder = st.empty()
contador_placeholder = st.empty()

# Lógica de captura en tiempo real
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara.")
        st.session_state.camera_active = False
    else:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo capturar el video.")
                break

            procesado, cantidad = detectar_personas(frame, modelo)
            rgb = cv2.cvtColor(procesado, cv2.COLOR_BGR2RGB)

            video_placeholder.image(rgb, channels="RGB", use_container_width=True)
            contador_placeholder.metric("👥 Personas detectadas", cantidad)

            time.sleep(0.05)

        cap.release()
        cv2.destroyAllWindows()
        st.info("Cámara detenida.")
