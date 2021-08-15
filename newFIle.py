import re
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook  # Google notebook

# Read the image
image = cv.imread("images/Straight Vertical/image0000.png")


def image_grey():
    # Image conversion to black and white
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Show the image
    plt.title("Image Grey Scaled")
    plt.imshow(grey, cmap='gray')
    plt.show()
    return grey


# The grey Image returned (calls function)
grey_Image = image_grey()


# Function to Blur the image
def image_blur():
    # Bluring the image
    Img_blur = cv.GaussianBlur(grey_Image, (5, 5), 0)

    plt.title("Image Blurred")
    plt.imshow(Img_blur, cmap='gray')
    plt.show()
    return Img_blur


# The blurred Image returned (calls function)
blur = image_blur()


# Canny Edge Detection - passing the blurred grey image
def image_canny():
    img_canny = cv.Canny(blur, 100, 200)
    plt.title("Canny Image")
    plt.imshow(img_canny, cmap='gray')
    plt.show()
    return img_canny


# Canny Image (calls function)
canny = image_canny()
