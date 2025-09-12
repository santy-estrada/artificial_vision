import cv2
import numpy as np
def get_image_size(image):
    # Get the size of the image
    return image.shape[:2]  # Returns (height, width)
def max_red(image,size):
    r_max=0
    for i in range (size[0]):
        for k in range (size[1]):
            analisis=image[i,k]
            if analisis[1]>r_max:
                r_max=analisis[1]
    return r_max

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
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)[1]
    mask_median= cv2.medianBlur(img_binary, 5)  #Params: (image, ksize --> must be odd)

    contours, hierarchy = cv2.findContours(mask_median, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #Params: (image, mode, method)
    
    contour_areas = [cv2.contourArea(item) for item in contours]
    max_area = max(contour_areas) if contour_areas else 0
    
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if x < 300 and testigo:
            if area == max_area:  # Filter out small contours
                # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Contours", frame)
                cv2.imshow("Mask Median", mask_median)
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
            flag = False
            if numCont == 2:
                cont1 += 1
            elif numCont == 3:
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