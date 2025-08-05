from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO entrenado
model = YOLO("best.pt")  # Asegúrate de que el modelo esté en el mismo directorio o da la ruta completa

# Ruta de la imagen que deseas predecir
image_path = r"D:\SumClasses\ArtVi\DeepLearning\g2\test\images\07-31-07_002_2_scale_2.5_aug_20250803_205344.jpg"  # Cambia esto por tu imagen

# Realizar la predicción
results = model(image_path)

# Mostrar resultados
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])  # clase predicha
        conf = box.conf[0]     # confianza
        x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja

        print(f"Clase: {cls}, Confianza: {conf:.2f}, BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Mostrar la imagen con las predicciones
result_image = results[0].plot()  # dibuja las cajas en la imagen original
cv2.imshow("Predicción YOLO - Nutrition", cv2.resize(result_image, (200, 700)))
cv2.waitKey(0)
cv2.destroyAllWindows()