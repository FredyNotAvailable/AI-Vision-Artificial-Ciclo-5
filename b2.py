import cv2
import tensorflow as tf
import numpy as np

# Ruta del modelo SSD MobileNet V2
MODEL_PATH = 'C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/MobileNetV2/ssd_mobilenet_v2_coco_2018_03_29/saved_model'

# Cargar modelo
detect_fn = tf.saved_model.load(MODEL_PATH).signatures['serving_default']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    h, w, _ = frame.shape
    person_count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:  # Solo clase 'person'
            person_count += 1
            ymin, xmin, ymax, xmax = boxes[i]
            start = (int(xmin * w), int(ymin * h))
            end = (int(xmax * w), int(ymax * h))
            cv2.rectangle(frame, start, end, (0, 255, 0), 2)
            cv2.putText(frame, f"Persona {person_count}", (start[0], start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar total de personas detectadas
    cv2.putText(frame, f"Personas detectadas: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Contador de Personas - SSD MobileNet V2", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
