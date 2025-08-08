import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/ResNet/faster_rcnn_resnet50_coco_2018_01_28/saved_model"

detect_fn = tf.saved_model.load(MODEL_PATH)
infer = detect_fn.signatures['serving_default']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = infer(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    person_count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:  # Solo 'person'
            person_count += 1

            ymin, xmin, ymax, xmax = boxes[i]
            start_point = (int(xmin * width), int(ymin * height))
            end_point = (int(xmax * width), int(ymax * height))

            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, f"Persona {person_count}", (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Personas detectadas: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Contador de Personas", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
