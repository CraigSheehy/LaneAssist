import re
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook  # Google notebook

# Read the image
image = cv.imread("images/Straight Vertical/image0000.png")
plt.imshow(image)
plt.show()


# Press 'Q' to close window
def quit(subject):
    if subject.key == 'q':
        plt.close()


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


# Function looks at just the white road lines.
# Using HSV Colourspace (Hue, Saturation, Value)
def identify_lines():
    original_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Identifying the white line using HSV placing into np.array
    upper_white = np.array([255, 7, 230], dtype=np.uint8)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)

    # Masking the white
    white_mask = cv.inRange(original_img, lower_white, upper_white)

    # A threshold to target only white colours in the image
    masked_image = cv.bitwise_and(grey_Image, white_mask)

    plt.imshow(masked_image)
    plt.title('Masked white image')
    plt.show()

    return masked_image


lane_lines = identify_lines()


# Declaring a region of interest within an image
def region_of_interest(placeholder):
    # Getting height and width of the image we want to use
    img_height = placeholder.shape[0]
    img_width = placeholder.shape[1]
    print("Image Height", img_height)
    print("Image Width", img_width, '\n')

    roi_triangle = np.array([
        [(0, img_height), (440, 216), (490, 216), (img_width, img_height)]
    ])

    # Creating a mask for the image
    img_mask = np.zeros_like(placeholder)

    # Combining the mask
    img_mask = cv.fillPoly(img_mask, roi_triangle, 255)
    img_mask = cv.bitwise_and(placeholder, img_mask)

    # Displaying
    plt.imshow(img_mask)
    plt.title("Masked Image")
    plt.show()

    return img_mask


roi = region_of_interest(lane_lines)


def hough_algorithm():
    # This is an array
    lane_lines_array = cv.HoughLinesP(roi, rho=2, theta=3.642 / 180, threshold=100, minLineLength=40, maxLineGap=5)

    # Displaying
    # print(type(lane_lines_array))
    # print(lane_lines_array)

    return lane_lines_array


lane_lines_array = hough_algorithm()


# This function averages the lines taken in from the hough_algorithm() function
# It finds the average slope and y intercepts per line segment
# Displaying one slide line
def average_of_lines():
    left_average = []
    right_average = []

    for line in lane_lines_array:
        # Array from hough transformation
        # print(line)
        x1, y1, x2, y2 = line.reshape(4)

        # print("x1: ", x1)
        # print("y1: ", y1)
        # print("x2: ", x2)
        # print("y2: ", y2)

        x1_array_numpy = np.array(x1)
        y1_array_numpy = np.array(y1)
        x2_array_numpy = np.array(x2)
        y2_array_numpy = np.array(y2)

        # Type = (numpy.ndarray)
        # print(type(x1_array))

        print("x1_array_numpy", x1_array_numpy)
        print("y1_array_numpy", y1_array_numpy)
        print("x2_array_numpy", x2_array_numpy)
        print("y2_array_numpy", y2_array_numpy)

        # Polynomial fit (finds the least square polynominal fit)
        # (best fitting curve to a given set of points)
        # using x1, x2, y1, y2
        polyfit_value = np.polyfit((x1, x2), (y1, y2), 1)
        print("polyfit_value = ", polyfit_value)
        print("polyfit_value[0] (slope_of_line_seg) = ", polyfit_value[0])
        print("polyfit_value[1] (y_intercept) = ", polyfit_value[1], '\n')

        slope_of_line_seg = polyfit_value[0]
        y_intercept = polyfit_value[1]

        if slope_of_line_seg < 0:
            left_average.append((slope_of_line_seg, y_intercept))
        else:
            right_average.append((slope_of_line_seg, y_intercept))

    # We expect left line to start with NEGATIVE slope
    print("Left Slope Average = ", left_average)
    print("Right Slope Average = ", right_average, '\n')



average_of_lines()
