'''
ACTIVIDAD:
- Contar número de piezas que pasan por la banda
- Clasificar las piezas en 2 grupos
- Contar el número de piezas de cada grupo
'''

#Mostrar opciones de procesar videos, avi, mp4, ip, webcam
import cv2
import functions as fn
import numpy as np

def max_red(image,size):
    r_max=0
    for i in range (size[0]):
        for k in range (size[1]):
            analisis=image[i,k]
            if analisis[1]>r_max:
                r_max=analisis[1]
    return r_max

# capture = cv2.VideoCapture(0)  # 0 for webcam. If USB place the id as int
capture = cv2.VideoCapture(r"imgs\video_1_12.avi")  # For video file (mp4 or avi)
#capture = cv2.VideoCapture("http://ip_address:port/video")  # For IP camera (protocol: rtsp, http, https, ftp + <camera ip>)

filtered_contours = []
flag = True

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    

    # Process the frame (for example, convert to grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)[1]
    maximo_detectado=max_red(frame,fn.get_image_size(frame))

    # Display the resulting frame
    # cv2.imshow('Video Frame', img_binary)
    
    #Contorno
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #Params: (image, mode, method)

    for contour in contours:
        area = cv2.contourArea(contour)
        if maximo_detectado > 50 and flag:  # Filter out small contours
            x,y,w,h = cv2.boundingRect(contour)
            # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow("Contours", frame)
            filtered_contours.append(contour)
            flag = False
                
        else:
            if maximo_detectado < 100:
                flag = True


    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print(f'Number of filtered contours found: {len(filtered_contours)}')