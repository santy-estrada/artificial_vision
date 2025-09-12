from ultralytics import YOLO
import cv2
import os

# Cargar el modelo YOLO entrenado
model = YOLO("bestProfe.pt")  # Asegúrate de que el modelo esté en el mismo directorio o da la ruta completa

cat = {0 : "car", 1: "truck"}
conteo = {"cars" : 0, "small trucks" : 0, "medium trucks": 0, "big trucks": 0}
sizes = [350, 530]

path = r"D:\SumClasses\ArtVi\DeepLearning\g2\eval_trucks_1"


# Realizar la predicción
for file_name in os.listdir(path):
    if file_name.lower().endswith(".jpg"):
        # Ruta de la imagen que deseas predecir
        image_path = os.path.join(path, file_name)
        # print(image_path)
        results = model(image_path)
        # Mostrar resultados
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])  # clase predicha
                conf = box.conf[0]     # confianza
                x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja

                print(f"Clase: {cls}, Confianza: {conf:.2f}, BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                size = y2 -y1
                
                if y2 >= 1000:
                    # print(f"Vehículo en fila baja tiene confianza: {conf:.2f}")
                    
                    if cls == 0:
                        # print(f"Vehiculo es {cat.get(0)}")
                        conteo["cars"] = conteo.get("cars") + 1
                        
                    elif cls == 1:
                        if size < sizes[0]:
                            # print(f"Vehículo en fila baja tiene confianza: {conf:.2f} y es pequeño")
                            conteo["small trucks"] = conteo.get("small trucks") + 1
                        elif size < sizes[1]:
                            # print(f"Vehículo en fila baja tiene confianza: {conf:.2f} y es mediano")
                            conteo["medium trucks"] = conteo.get("medium trucks") + 1
                        else:
                            # print(f"Vehículo en fila baja tiene confianza: {conf:.2f} y es grande")
                            conteo["big trucks"] = conteo.get("big trucks") + 1
                            
            for c,v in conteo.items():
                print(str(c) + ": " + str(v))
            print("--------------")
            # Mostrar la imagen con las predicciones
            result_image = results[0].plot()  # dibuja las cajas en la imagen original
            cv2.imshow("Predicción YOLO - Nutrition", cv2.resize(result_image, (200, 700)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()