import functions as fn
import cv2

window = "Reto2"

green_range = ((38, 115), (81, 255), (14, 255))  # HSV range for green color
blue_range = ((80, 128), (58, 224), (24, 255))   # HSV range for blue color
yellow_range = ((20, 119), (14, 255), (38, 228)) # HSV range for yellow color
red_range = ((0, 186), (0, 255), (62, 255))      # HSV range for red color

def main():
    image_path = "imgs/good_yellow.png"  # Replace with your image path
    imageBgr, imageGray = fn.read_image(image_path)
    
    imageBgr = fn.resize_image(imageBgr)
    
    imageHSV = fn.transformSpaceBGR2HSV(imageBgr)
    
    while True:
        vals = fn.get_trackbar_values("Binary Image", type=2)
        lowLim = vals[:3]
        upLim = vals[3:]
        binPic = fn.binary(imageHSV, method=2, rgbMin=lowLim, rgbMax=upLim)

        binPic = fn.binary(imageHSV, method=2, rgbMin=(80, 58, 24), rgbMax=(128, 224, 255))

        contours, _ = cv2.findContours(binPic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"Contour area: {area}")

        fn.show_image(binPic, title="Binary Image", type=1)
        size = fn.get_image_size(binPic)
        cont = 0
        
        for i in range(size[0]):
            for j in range(size[1]):
                if binPic[i, j] > 0:
                    cont += 1
                    
        
        # print(cont)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
   

if __name__ == "__main__":
    fn.create_trackbar("Binary Image", type=2)
    cv2.waitKey(10)  # Ensure the trackbar is created before proceeding
    
    main()
    print("Program completed successfully.")