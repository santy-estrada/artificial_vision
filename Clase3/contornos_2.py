'''
List --> Contornos sin jerarquía
Tree --> Contornos con jerarquía
External --> Contornos más externos
'''

import cv2
import functions as fn
import numpy as np

def main():
    image = cv2.imread("imgs/leukemia.jpg")
    maskColor = np.zeros(image.shape, dtype="uint8")

    upper_bound = (246, 91, 255)
    lower_bound = (107, 0, 104)

    #Create a binary mask
    mask = fn.binary(image, method=2, rgbMin=lower_bound, rgbMax=upper_bound)


    #Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #Params: (image, mode, method)
    print(f'Number of contours found: {len(contours)}')
    filtered_contours = []
    #Analyze contours
    if(len(contours) > 0):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 40:
                cv2.drawContours(maskColor, cnt, -1, (0, 0, 255), 3)  #Fill contour on maskColor
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)  #Draw rectangle around first contour found
                fn.show_image(image, title="Rectangle", type=0)
                fn.show_image(maskColor, title="Mask Color", type=0)
                filtered_contours.append(cnt)
        print(f'Number of filtered contours found: {len(filtered_contours)}')

    #Draw contours on the original image
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 4)   #Params: (image, contours, contourIdx, color, thickness)

    #Show images
    fn.show_image(mask, title="Mask", type=1)
    fn.show_image(image, title="Contours", type=1)
    
    cv2.waitKey(0)          # Wait for a key press if 0 or wait for specified milliseconds
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
    print("Program completed successfully.")