from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo YOLOv8
model = YOLO(r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\bestPlateCar.pt')

# Ruta del video de entrada y de salida
video_path = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\Videos\Video_deteccion.mp4'
output_video_path = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\Videos\Video_carros_1_predicted.mp4'

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

    # Extraer las cajas delimitadoras (bounding boxes) y dibujar rectángulos
    for result in results:
        for box in result.boxes:  # Para cada caja delimitadora
            # Obtener las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            print(f"Confianza de detección: {confidence:.2f}")

            if confidence > 0.4:
                # Dibujar el rectángulo alrededor del objeto detectado
                cv2.rectangle(frame, (x1 - 5, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2)
                label = f'Placa ({confidence:.2f})'

                # Poner la etiqueta con el ID de la clase y la confianza
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                imgRoi = frame[y1 - 3:y2 + 3, x1 - 5:x2 + 3]

                if imgRoi.size != 0 and imgRoi.shape[0] > 0 and imgRoi.shape[1] > 0:
                    
                    # 1. Convertir ROI a escala de grises
                    gray_roi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
                    
                    # 2. Aplicar umbralización de Otsu para binarizar la imagen.
                    _, imgBinary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # 3. Limpiar la máscara para eliminar ruido
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(imgBinary, cv2.MORPH_OPEN, kernel)
                    
                    # 4. Encontrar TODOS los contornos
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Crear una copia de la imagen ROI para dibujar los contornos
                    roi_with_contours = imgRoi.copy()

                    # 5. Iterar y filtrar los contornos basándose en su forma
                    for cnt in contours:
                        # Obtener el rectángulo delimitador del contorno
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # --- FILTRO MEJORADO BASADO EN LA FORMA DE UN CARACTER ---
                        # Se usan propiedades geométricas para decidir si un contorno es un caracter.
                        
                        # a) Relación de aspecto (aspect ratio): ancho / alto.
                        #    Los caracteres suelen ser más altos que anchos.
                        aspect_ratio = w / h
                        
                        # b) Altura del contorno: debe ser una fracción significativa de la altura de la placa.
                        #    Esto evita contornos muy pequeños (ruido) o muy grandes.
                        
                        # c) Área: para descartar contornos demasiado pequeños.
                        area = cv2.contourArea(cnt)
                        
                        # CONDICIONES (estos valores son un punto de partida, ajústalos si es necesario):
                        # - El aspect ratio debe estar entre 0.2 y 1.0 (más alto que ancho).
                        # - La altura 'h' debe ser al menos el 30% de la altura de la placa.
                        # - El área debe ser mayor a un umbral para evitar ruido.
                        if (0.2 < aspect_ratio < 1.0) and (h > 0.3 * imgRoi.shape[0])and (h < 0.6 * imgRoi.shape[0]) and (area > 30) and (area < 800):
                            # Si el contorno cumple las condiciones, es probable que sea un caracter.
                            # Dibujarlo en la imagen.
                            cv2.drawContours(roi_with_contours, [cnt], -1, (0, 0, 255), 2) # Dibujar en rojo

                    # 6. Mostrar la imagen con los contornos de cada caracter
                    cv2.imshow("Caracteres Detectados", roi_with_contours)
                    cv2.imshow("Mascara Binaria de Caracteres", mask)

    # Escribir el frame anotado en el video de salida
    # out.write(frame)
    cv2.imshow("predict video", cv2.resize(frame, (round(width * 0.5), round(height * 0.5))))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()

cv2.destroyAllWindows()
