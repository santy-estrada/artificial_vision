import cv2
import numpy as np


filtered_contours = []
cont1 = 0
cont2 = 0
flag = False
capture = cv2.VideoCapture(r'imgs\video_1_12.avi') #aqui metes la dirección del video
testigo=True
while capture.isOpened():   
    ret, frame = capture.read() #lee el video  frame es la imagen actual
    if not ret:
        print("No se pudo capturar el video")
        break
    # cv2.imshow('frame', frame)
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_binary = cv2.inRange(hsv_frame, (80, 0, 40), (115, 255, 255))
    img_binary= cv2.medianBlur(img_binary, 5)  #Params: (image, ksize --> must be odd)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #Params: (image, mode, method)
    
    contour_areas = [cv2.contourArea(item) for item in contours]
    max_area = max(contour_areas) if contour_areas else 0
    max_area_contour = contours[contour_areas.index(max_area)] if contour_areas else None
    x,y,w,h = cv2.boundingRect(max_area_contour) if max_area_contour is not None else (0,0,0,0)

    # cv2.imshow("Contours", frame)
    # cv2.imshow("Mask Median", img_binary)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if x < 300 and testigo:
            if area == max_area and area > 5000:  # Filter out small contours
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Contours", frame)
                cv2.imshow("Mask Median", img_binary)
                filtered_contours.append(contour)
                print(f'Area del contorno: {area}')
                # print(f'x: {x}, y: {y}, w: {w}, h: {h}')
                testigo = False
                flag = True
                    
        elif x > 400:
            testigo = True
            
        if flag:
            numCont = len(contours)
            print(f'Number of contours found: {numCont}')
            large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
            print(f'Contours with area > 5000: {len(large_contours)}\n')
            flag = False
            if len(large_contours) == 2:
                cont1 += 1
            elif len(large_contours) == 3:
                cont2 += 1
    

    if cv2.waitKey(10) & 0xFF == ord('q'): #con el waitKey puedes poner la velocidad del video
        break
print(f'Number of filtered contours found: {len(filtered_contours)}')
print(f'Number of pieces in group 1 (circle): {cont1}')
print(f'Number of pieces in group 2 (eigth): {cont2}')
#contar piezas que pasan por la banda

#calsificar las piezas en 2 grupos
#contar cuantas piezas hay en cada grupo
capture.release()
cv2.destroyAllWindows()