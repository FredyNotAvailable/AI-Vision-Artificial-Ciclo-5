import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Reducir logs

import streamlit as st
import cv2
import numpy as np
import time
from yourmodel import CameraStream, load_detector, detectar_personas

st.title("🎥 Detector de Personas en vivo")

CAM_OPTIONS = [0, 1]
cam_index = st.selectbox("Selecciona la cámara", CAM_OPTIONS)

if "stream" not in st.session_state:
    st.session_state.stream = None
if "infer" not in st.session_state:
    st.session_state.infer = load_detector()
if "running" not in st.session_state:
    st.session_state.running = False

start_button = st.button("▶️ Iniciar")
stop_button = st.button("⏹ Detener")

video_placeholder = st.empty()
count_placeholder = st.empty()

if start_button and not st.session_state.running:
    try:
        st.session_state.stream = CameraStream(cam_index=cam_index).start()
        st.session_state.running = True
    except RuntimeError as e:
        st.error(f"No se pudo abrir la cámara {cam_index}: {e}")

if stop_button and st.session_state.running:
    st.session_state.running = False
    if st.session_state.stream:
        st.session_state.stream.stop()
        st.session_state.stream = None

while st.session_state.running:
    frame = st.session_state.stream.read()
    if frame is not None:
        frame_proc, count = detectar_personas(frame, st.session_state.infer)
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True)
        count_placeholder.metric("Personas detectadas", count)
    else:
        time.sleep(0.01)
    # Esto permite a Streamlit actualizar la UI en cada iteración
    if not st.session_state.running:
        break
    time.sleep(0.001)
