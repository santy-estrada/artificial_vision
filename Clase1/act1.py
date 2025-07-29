import functions as funcs
import numpy as np

def main():
    image_path = "imgs/Kpi.jpg"  # Replace with your image path
    imageBgr, imageGray = funcs.read_image(image_path)
    
    funcs.show_image(imageBgr, "Original BGR Image")
    # funcs.show_image(imageGray, "Original Grayscale Image")
    
    #Punto 1
    color = input("Enter color (green, blue, red) to modify pixel values: ")
    
    newImg = funcs.turn_color(imageBgr, color)
    funcs.show_image(newImg, "Modified Color Image")
    
    #Punto 2
    invertedY = funcs.invert_y(imageBgr)
    funcs.show_image(invertedY)
    
    #Punto 3
    invertedX = funcs.invert_x(imageBgr)
    funcs.show_image(invertedX, "Inverted Image X")
    
    #Punto 4
    gridImg = funcs.cuadricula(imageBgr)
    funcs.show_image(gridImg, "Image with Grid")
    
    #Reto
    n = int(input("Enter grid size (n): "))
    bgrPattern = funcs.cuadricula(imageBgr, n)
    funcs.show_image(bgrPattern, "BGR Pattern Image")
    
    bgrXPattern = funcs.patternX(imageBgr, n)
    funcs.show_image(bgrXPattern, "BGR X Pattern Image")
    
    bgrYPattern = funcs.patternY(imageBgr, n)
    funcs.show_image(bgrYPattern, "BGR Y Pattern Image")


if __name__ == "__main__":
    main()
    print("Program completed successfully.")