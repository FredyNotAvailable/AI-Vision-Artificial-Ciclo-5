import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

st.set_page_config(layout="centered", page_title="Detector de Personas - Biblioteca UIDE")

st.title("🎥 Detector de Personas en la Biblioteca - UIDE")
st.write("Esta aplicación detecta y cuenta personas en tiempo real usando modelos de visión artificial.")

# Cargar modelos de detección
MODEL_PATH_FRCNN = "C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/ResNet/faster_rcnn_resnet50_coco_2018_01_28/saved_model"
MODEL_PATH_MOBILENET = "C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/MobileNetV2/ssd_mobilenet_v2_coco_2018_03_29/saved_model"

detect_fn_frcnn = tf.saved_model.load(MODEL_PATH_FRCNN).signatures['serving_default']
detect_fn_mobilenet = tf.saved_model.load(MODEL_PATH_MOBILENET).signatures['serving_default']

LABELS = {1: "person"}

# Funciones de detección
def detectar_personas(imagen_bgr, modelo):
    input_tensor = tf.convert_to_tensor([imagen_bgr], dtype=tf.uint8)
    detections = modelo(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = imagen_bgr.shape
    person_count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:
            person_count += 1
            ymin, xmin, ymax, xmax = boxes[i]
            start = (int(xmin * width), int(ymin * height))
            end = (int(xmax * width), int(ymax * height))
            cv2.rectangle(imagen_bgr, start, end, (0, 255, 0), 2)
            cv2.putText(imagen_bgr, f"Persona {person_count}", (start[0], start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return imagen_bgr, person_count

# --- Interfaz cámara en tiempo real ---
st.subheader("🧠 Modo de detección")

modelo_seleccionado = st.radio("Seleccione el modelo:", [
    "Faster R-CNN (ResNet50)",
    "SSD MobileNetV2"
])

modelo = detect_fn_frcnn if modelo_seleccionado == "Faster R-CNN (ResNet50)" else detect_fn_mobilenet

if st.button("📷 Activar cámara"):
    cap = cv2.VideoCapture(1)
    video_placeholder = st.empty()
    contador_placeholder = st.empty()

    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara.")
    else:
        st.success("Cámara activada. Presiona ESC para detener.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo capturar el video.")
                    break

                procesado, cantidad = detectar_personas(frame, modelo)
                rgb = cv2.cvtColor(procesado, cv2.COLOR_BGR2RGB)

                video_placeholder.image(rgb, channels="RGB", use_column_width=True)
                contador_placeholder.metric("👥 Personas detectadas", cantidad)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                time.sleep(0.05)
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            st.info("Cámara detenida.")

