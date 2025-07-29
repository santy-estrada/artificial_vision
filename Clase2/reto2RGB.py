import functions as fn
import cv2

window = "Reto2"

def main():
    image_path = "imgs/imagenFachada.png"  # Replace with your image path
    imageBgr, imageGray = fn.read_image(image_path)
    
    imageBgr = fn.resize_image(imageBgr)
    
    imageRGB = fn.transformSpaceBGR2RGB(imageBgr)
    
    old_color_min = (10, 130, 125)
    old_color_max = (30, 255, 255)

    # fn.show_image(imageRGB, title="Image RGB")

    while True:
        vals = fn.get_trackbar_values("Binary Image", type=1)
        lowLim = vals[:3]
        upLim = vals[3:]
        binPic = fn.binary(imageRGB, method=2, rgbMin=lowLim, rgbMax=upLim)
        fn.show_image(binPic, title="Binary Image", type=1)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            old_color_min = lowLim
            old_color_max = upLim
            print(f"Old color min set to: {old_color_min}")
            print(f"Old color max set to: {old_color_max}")

        val = fn.get_trackbar_values(window=window, type=3, name="Color")
        if val[0] == 0:
            test = fn.replace_color(imageBgr, old_color_min=old_color_min, old_color_max=old_color_max, new_color=(0, 0, 255), color_space="RGB")  # Red in RGB
        elif val[0] == 1:
            test = fn.replace_color(imageBgr, old_color_min=old_color_min, old_color_max=old_color_max, new_color=(0, 255, 0), color_space="RGB")  # Green in RGB
        else:
            test = fn.replace_color(imageBgr, old_color_min=old_color_min, old_color_max=old_color_max, new_color=(255, 0, 0), color_space="RGB")  # Blue in RGB

        fn.drawRectangle(test, x=110, y=10, width=70, height=30, color=(0, 0, 255), filled=True)  # Example rectangle
        fn.drawRectangle(test, x=350, y=10, width=70, height=30, color=(0, 255, 0), filled=True)  # Example rectangle
        fn.drawRectangle(test, x=600, y=10, width=70, height=30, color=(255, 0, 0), filled=True)  # Example rectangle
    
        fn.show_image(test, title=window, type=1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
   

if __name__ == "__main__":
    fn.create_trackbar("Binary Image", type=1)
    fn.create_trackbar(window, type=3, name = "Color", max_value=2)
    
    cv2.waitKey(10)  # Ensure the trackbar is created before proceeding

    main()
    print("Program completed successfully.")