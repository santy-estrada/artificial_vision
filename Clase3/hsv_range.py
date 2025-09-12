import functions as fn
import cv2

window = "Reto2"

def main():
    image_path = "imgs/img.png"  # Replace with your image path
    imageBgr, imageGray = fn.read_image(image_path)
    
    imageBgr = fn.resize_image(imageBgr)
    
    imageHSV = fn.transformSpaceBGR2HSV(imageBgr)
    
    while True:
        vals = fn.get_trackbar_values("Binary Image", type=2)
        lowLim = vals[:3]
        upLim = vals[3:]
        binPic = fn.binary(imageHSV, method=2, rgbMin=lowLim, rgbMax=upLim)
        
        # binPic = fn.binary(imageHSV, method=2, rgbMin=(125, 250, 110), rgbMax=(145, 255, 140))

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