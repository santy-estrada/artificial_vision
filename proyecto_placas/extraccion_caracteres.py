from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# Cargar el modelo YOLOv8
model = YOLO(r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\bestPlateCar.pt')

# Ruta del video de entrada y de salida
video_path = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\Videos\Video_deteccion.mp4'
output_video_path = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\Videos\Video_carros_1_predicted.mp4'

# --- CONFIGURACIÓN PARA GUARDAR CARACTERES ---
# Directorio principal donde se guardará el dataset
DATASET_DIR = "Proyecto_deteccion_placas\dataset_caracteres"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    print(f"Directorio para el dataset creado en: {DATASET_DIR}")

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Obtener la resolución del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear un objeto para escribir el video con las predicciones
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video de salida
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Procesar cada frame del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no quedan más frames

    # Realizar la predicción en el frame actual
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # --- NUEVO: Bandera para saltar el frame ---
    skip_frame = False

    # Extraer las cajas delimitadoras (bounding boxes) y dibujar rectángulos
    for result in results:
        for box in result.boxes:  # Para cada caja delimitadora
            # Obtener las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            
            if confidence > 0.4:
                # Dibujar el rectángulo alrededor del objeto detectado
                cv2.rectangle(frame, (x1 - 5, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2)
                label = f'Placa ({confidence:.2f})'

                # Poner la etiqueta con el ID de la clase y la confianza
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                imgRoi = frame[y1 - 3:y2 + 3, x1 - 5:x2 + 3]

                if imgRoi.size != 0 and imgRoi.shape[0] > 0 and imgRoi.shape[1] > 0:
                    
                    gray_roi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
                    _, imgBinary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(imgBinary, cv2.MORPH_OPEN, kernel)
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    roi_with_contours = imgRoi.copy()

                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = w / h
                        area = cv2.contourArea(cnt)

                        # MODIFICAR

                        if (0.2 < aspect_ratio < 1.0) and (h > 0.3 * imgRoi.shape[0]) and (area > 30) and (area < 800):
                            cv2.drawContours(roi_with_contours, [cnt], -1, (0, 0, 255), 2)

                            # --- LÓGICA PARA GUARDAR CARACTERES ---
                            char_image = mask[y:y+h, x:x+w]
                            TARGET_SIZE = (20, 20)
                            char_resized = cv2.resize(char_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                            cv2.imshow("Guardar Caracter", char_resized)
                            print("Presiona la tecla del caracter. 'ESC' para ignorar. ' ' para saltar el frame.")
                            
                            key = cv2.waitKey(0) & 0xFF 

                            if key == 27: # Tecla ESC
                                print("Caracter ignorado.")
                                cv2.destroyWindow("Guardar Caracter")
                                continue
                            
                            # --- NUEVA OPCIÓN PARA SALTAR EL FRAME ---
                            elif key == ord(' '):
                                print("Saltando el resto de caracteres de este frame...")
                                skip_frame = True
                                cv2.destroyWindow("Guardar Caracter")
                                break # Rompe el bucle de los contornos (caracteres)
                            
                            elif key != 255: # Si se presionó una tecla válida
                                char = chr(key).upper() 
                                char_folder_path = os.path.join(DATASET_DIR, char)
                                if not os.path.exists(char_folder_path):
                                    os.makedirs(char_folder_path)
                                
                                count = len(os.listdir(char_folder_path))
                                file_path = os.path.join(char_folder_path, f"{count + 1}.png")
                                cv2.imwrite(file_path, char_resized)
                                
                                print(f"Caracter '{char}' guardado en: {file_path}")
                                cv2.destroyWindow("Guardar Caracter")

                    cv2.imshow("Caracteres Detectados", roi_with_contours)
                    cv2.imshow("Mascara Binaria de Caracteres", mask)
            
            # Si la bandera de saltar se activó, rompemos también este bucle
            if skip_frame:
                break
        
        # Y finalmente rompemos el bucle principal de resultados
        if skip_frame:
            break

    # Escribir el frame anotado en el video de salida
    # out.write(frame)
    cv2.imshow("predict video", cv2.resize(frame, (round(width * 0.5), round(height * 0.5))))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video procesado y guardado en {output_video_path}")