'''
List --> Contornos sin jerarquía
Tree --> Contornos con jerarquía
External --> Contornos más externos
'''

import cv2
import functions as fn

def main():
    image = cv2.imread("imgs/leukemia.jpg")

    upper_bound = (246, 91, 255)
    lower_bound = (107, 0, 104)

    #Create a binary mask
    mask = fn.binary(image, method=2, rgbMin=lower_bound, rgbMax=upper_bound)
    
    
    #Espatial Filters
    '''
    Los filtros buscan suavizar la imagen y eliminar el ruido. Pixel central parecido a pixeles vecinos
    '''
    mask_median= cv2.medianBlur(mask, 3)  #Params: (image, ksize --> must be odd)
    #2D filters and Gaussian
    mask_gaussian= cv2.GaussianBlur(mask, (5, 5), 0)  #Params: (image, ksize, sigmaX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_2d = cv2.filter2D(mask, -1, kernel)  #Params: (image, ddepth, kernel)
    
    
    #Morphological Filters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   #Params: (shape --> Ellipse | RECT, ksize)
    #Erosion
    mask_eroded = cv2.erode(mask, kernel, iterations=1)  #Params: (image, kernel, iterations)
    #Dilation
    mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)  #Params: (image, kernel, iterations)
    #Closing
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #Aperture
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)


    #Find contours in the mask
    contours, _ = cv2.findContours(mask_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #Params: (image, mode, method)
    print(f'Number of contours found: {len(contours)}')
    #Analyze contours
    if(len(contours) > 0):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)  #Draw rectangle around first contour found
            fn.show_image(image, title="Rectangle", type=0)
        
       
    #Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 4)   #Params: (image, contours, contourIdx, color, thickness)

    
    #Show images
    fn.show_image(mask, title="Mask", type=1)
    fn.show_image(mask_median, title="Mask Median", type=1)
    # fn.show_image(mask_gaussian, title="Mask Gaussian", type=1)
    # fn.show_image(mask_2d, title="Mask 2D Filter", type=1)
    # fn.show_image(mask_eroded, title="Mask Eroded", type=1)
    # fn.show_image(mask_dilated, title="Mask Dilated", type=1)
    # fn.show_image(mask_closed, title="Mask Closed", type=1)
    # fn.show_image(mask_opened, title="Mask Opened", type=1)

    fn.show_image(image, title="Contours", type=1)
    
    cv2.waitKey(0)          # Wait for a key press if 0 or wait for specified milliseconds
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
    print("Program completed successfully.")