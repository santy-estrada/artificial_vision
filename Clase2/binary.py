import cv2
import functions as fn

 
def main():
    image_path = "imgs/5.jpg"  # Replace with your image path
    # image_path = "imgs/contraste.png"
    imageBgr, imageGray = fn.read_image(image_path)
    
    imageBgr = fn.resize_image(imageBgr)
    imageGray = fn.resize_image(imageGray)
    imageHSV = fn.transformSpaceBGR2HSV(imageBgr)

    while True:
        #get trackbar values
        vals = fn.get_trackbar_values("Binary Image", type=2)

        imageBinary = fn.binary(imageHSV, method=2, rgbMin=vals[:3], rgbMax=vals[3:])

        # fn.show_image(imageHSV, title="Image HSV", type=1)
        # fn.show_image(imageGray, title="Grayscale Image")
        fn.show_image(imageBinary, title="Binary Image", type=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    fn.create_trackbar("Binary Image", type=2)
    main()
    print("Program completed successfully.")
