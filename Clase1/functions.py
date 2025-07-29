import cv2
import numpy as np

def read_image(image_path):
    """
    Reads an image from the specified path and returns it.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The image read from the file.
    """
    imageBgr = cv2.imread(image_path, 1)      #flag = 1 (default), bgr; flag = 0, grayscale
    imageGray = cv2.imread(image_path, 0)          #flag = 0, grayscale
    
    return imageBgr, imageGray

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

def turn_color(image, color):
    """
    Only keeps the specified color channel in a BGR image.
    
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
            if color == 'blue':
                modify_pixel_value(img_copy, j, i, (b, 0, 0))
            elif color == 'green':
                modify_pixel_value(img_copy, j, i, (0, g, 0))
            elif color == 'red':
                modify_pixel_value(img_copy, j, i, (0, 0, r))
                
    return img_copy

def invert_y(image):
    """
    Inverts image with respect to Y

    Args:
        image (numpy.ndarray): The BGR image to modify.

    Returns:
        numpy.ndarray: The modified inverted image.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    newImg = np.empty_like(img_copy)

    for i in range(size[0]):
        for j in range(size[1]):
            newImg[i,j] = get_pixel_value(img_copy, size[0] - j - 1, i)
            
    return newImg

def invert_x(image):
    """
    Inverts image with respect to X

    Args:
        image (numpy.ndarray): The BGR image to modify.

    Returns:
        numpy.ndarray: The modified inverted image.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    newImg = np.empty_like(img_copy)

    for i in range(size[0]):
        for j in range(size[1]):
            newImg[i,j] = get_pixel_value(img_copy, j, size[1] - 1 - i)
            
    return newImg

def cuadricula(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    if n <= 0:
        Ydis = size[0] / 2 if size[0] % 2 == 0 else (size[0] // 2) + 1
        Xdis = size[1] / 2 if size[1] % 2 == 0 else (size[1] // 2) + 1
        for i in range(size[0]):
            for j in range(size[1]):
                b,g,r = get_pixel_value(img_copy, j, i)
                if i < Ydis and j >= Xdis:
                    modify_pixel_value(img_copy, j, i, (b, 0, 0))
                elif i >= Ydis and j < Xdis:
                    modify_pixel_value(img_copy, j, i, (0, g, 0))
                elif i >= Ydis and j >= Xdis:
                    modify_pixel_value(img_copy, j, i, (0, 0, r))
    else:       
        
        # n = 4, size = 8x8
        # For i in range(8): (rows)
        #   For j in range(8): (cols)
        #       block_x = j // 4  # 0 for j=0..3, 1 for j=4..7
        #       block_y = i // 4  # 0 for i=0..3, 1 for i=4..7
        #       c = (block_x + block_y) % 3

        # Block positions:
        # (block_y, block_x): (0,0), (0,1), (1,0), (1,1)
        # c values: (0+0)%3=0 (blue), (0+1)%3=1 (green), (1+0)%3=1 (green), (1+1)%3=2 (red)

        # So:
        # Top-left 4x4: blue
        # Top-right 4x4: green
        # Bottom-left 4x4: green
        # Bottom-right 4x4: red

        # For each pixel in each block, only the corresponding color channel is kept.

        for i in range(size[0]): 
            for j in range(size[1]):  
                b, g, r = get_pixel_value(img_copy, j, i)
    
                # Determine current block position
                block_x = j // n
                block_y = i // n

                # Choose which color to keep
                c = (block_x + block_y) % 3

                if c == 0:
                    modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
                elif c == 1:
                    modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
                elif c == 2:
                    modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                    
                

    return img_copy

def patternX(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)

    for i in range(size[0]): 
        for j in range(size[1]):  
            b, g, r = get_pixel_value(img_copy, j, i)

            # Determine current block position
            block_x = j // n

            # Choose which color to keep
            c = (block_x) % 3

            if c == 0:
                modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
            elif c == 1:
                modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
            elif c == 2:
                modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                
            

    return img_copy
    
def patternY(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)

    for i in range(size[0]): 
        for j in range(size[1]):  
            b, g, r = get_pixel_value(img_copy, j, i)

            # Determine current block position
            block_y = i // n

            # Choose which color to keep
            c = (block_y) % 3

            if c == 0:
                modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
            elif c == 1:
                modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
            elif c == 2:
                modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                
            

    return img_copy
