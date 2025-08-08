import cv2
import numpy as np
import tensorflow as tf

# Rutas de los modelos
MODEL_PATH_FRCNN = "C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/ResNet/faster_rcnn_resnet50_coco_2018_01_28/saved_model"
MODEL_PATH_MOBILENET = "C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/MobileNetV2/ssd_mobilenet_v2_coco_2018_03_29/saved_model"

# Cargar modelos una sola vez
detect_fn_frcnn = tf.saved_model.load(MODEL_PATH_FRCNN).signatures['serving_default']
detect_fn_mobilenet = tf.saved_model.load(MODEL_PATH_MOBILENET).signatures['serving_default']

LABELS = {1: "person"}

def get_model(name: str):
    """Retorna el modelo de detección según el nombre."""
    if name == "Faster R-CNN (ResNet50)":
        return detect_fn_frcnn
    elif name == "SSD MobileNetV2":
        return detect_fn_mobilenet
    else:
        raise ValueError("Modelo no reconocido.")

def detectar_personas(imagen_bgr, modelo):
    """Detecta personas en una imagen y devuelve la imagen procesada y el conteo."""
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
