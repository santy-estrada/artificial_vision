import cv2
import functions as funcs
print("OpenCV version:", cv2.__version__)

# def read_image(image_path):
#     """
#     Reads an image from the specified path and returns it.
    
#     Args:
#         image_path (str): The path to the image file.
        
#     Returns:
#         numpy.ndarray: The image read from the file.
#     """
#     imageBgr = cv2.imread(image_path, 1)      #flag = 1 (default), bgr; flag = 0, grayscale
#     imageGray = cv2.imread(image_path, 0)          #flag = 0, grayscale
    
#     return imageBgr, imageGray

def show_image(image, title="Image"):
    """
    Displays an image in a window.
    
    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)          # Wait for a key press if 0 or wait for specified milliseconds
    cv2.destroyAllWindows()
    
def resize_image(image):
    """
    Resizes an image.
    
    Args:
        image (numpy.ndarray): The image to resize.
        
    Returns:
        numpy.ndarray: The resized image.
    """
    resized_image = cv2.resize(image, (640, 480))
    return resized_image

def get_image_size(image):
    """
    Returns the size of the image.
    
    Args:
        image (numpy.ndarray): The image whose size is to be returned.
        
    Returns:
        tuple: The dimensions of the image (height, width).
    """
    return image.shape[:2]  # Returns (height, width) for BGR or Grayscale images

def get_pixel_value(image, x, y):
    """
    Returns the pixel value at the specified coordinates.
    
    Args:
        image (numpy.ndarray): The image from which to get the pixel value.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        
    Returns:
        tuple: The pixel value at (x, y) for BGR images or a single value for Grayscale images.
    """
    return image[y, x]  # Note: OpenCV uses (y, x) indexing

def modify_pixel_value(image, x, y, value):
    """
    Modifies the pixel value at the specified coordinates.
    
    Args:
        image (numpy.ndarray): The image in which to modify the pixel value.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        value (tuple or int): The new pixel value (BGR tuple for color images, single int for grayscale).
    """
    image[y, x] = value  # Note: OpenCV uses (y, x) indexing
    
def gray_to_black_and_white(image):
    """
    Converts a grayscale image to black and white.
    
    Args:
        image (numpy.ndarray): The grayscale image to convert.
        
    Returns:
        numpy.ndarray: The black and white image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image is not a grayscale image.")
    
    img_copy = image.copy()
    gray_size = get_image_size(img_copy)
    for i in range(gray_size[0]):
        for j in range(gray_size[1]):
            if get_pixel_value(img_copy, j, i) < 128:
                modify_pixel_value(img_copy, j, i, 0)
            else:
                modify_pixel_value(img_copy, j, i, 255)
                
    return img_copy

def invert_B_R(image):
    """
    Inverts the Blue and Red channels of a BGR image.
    
    Args:
        image (numpy.ndarray): The BGR image to modify.
        
    Returns:
        numpy.ndarray: The modified image with inverted Blue and Red channels.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image is not a BGR image.")
    
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    for i in range(size[0]):
        for j in range(size[1]):
            b,g,r = get_pixel_value(img_copy, j, i)
            modify_pixel_value(img_copy, j, i, (r, g, b))
                
    return img_copy


def main():
    image_path = "imgs/img1.png"  # Replace with your image path
    imageBgr, imageGray = funcs.read_image(image_path)
    
    if imageBgr is not None:
        print("Image read successfully in BGR format.")
        resized_image = resize_image(imageBgr)

    else:
        print("Failed to read the image in BGR format.")
    
    if imageGray is not None:
        print("Image read successfully in Grayscale format.")
        resized_gray_image = resize_image(imageGray)
        
    else:
        print("Failed to read the image in Grayscale format.")
        
    print("Image size (BGR):", get_image_size(imageBgr))
    bgr_size = get_image_size(imageBgr)
    print("Image size (Grayscale):", bgr_size)
    
    show_image(invert_B_R(imageBgr), "Inverted BGR Image")
    
    print("Image size after resizing (BGR):", get_image_size(resized_image))
    gray_size = get_image_size(resized_gray_image)
    print("Image size after resizing (Grayscale):", gray_size)        

    show_image(gray_to_black_and_white(resized_gray_image), "Modified Resized Grayscale Image")
    
    value_pixel_bgr = get_pixel_value(imageBgr, 100, 100)  # Example coordinates
    value_pixel_gray = get_pixel_value(imageGray, 100, 100)  # Example coordinates
    
    print("Pixel value at (100, 100) in BGR image:", value_pixel_bgr)
    print("Blue value at (100, 100):", value_pixel_bgr[0])  # BGR format
    print("Green value at (100, 100):", value_pixel_bgr[1])  # BGR format
    print("Red value at (100, 100):", value_pixel_bgr[2])  # BGR format
    
    print("Pixel value at (100, 100) in Grayscale image:", value_pixel_gray)   
        

    show_image(imageGray, "Grayscale Image")
    show_image(resized_gray_image, "Resized Grayscale Image")
    
    show_image(imageBgr, "BGR Image")   
    show_image(resized_image, "Resized BGR Image")
        
        

if __name__ == "__main__":
    main()
    print("Program completed successfully.")