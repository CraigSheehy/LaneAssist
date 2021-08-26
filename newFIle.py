import glob
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def read_images():
    global name
    path = 'images/Busy Town/'

    for name in os.listdir(path):
        if name.endswith(".png") or name.endswith(".jpg"):
            # name.__str__()
            # print(name)

            # Read the image
            image = 'images/Busy Town/' + name  # cv.imread("images/Straight Vertical/" + name)

            array_of_image_urls = [image]
            # print(array_of_image_urls)
            image = cv.imread(image)
            cv.imshow("Raw Stock Image", image)
            break

            # Destroy window when pressing Q
            cv.waitKey(0)  # & 0xFF == ord('q')
            cv.destroyAllWindows()
        else:
            continue
    return image


def create_video():
    img_array = []
    for filename in glob.glob('images/Busy Town/*.png'):
        img = cv.imread(filename)
        H, W, layers = img.shape
        size = (W, H)
        img_array.append(img)

    video_output = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        video_output.write(img_array[i])
    video_output.release()


create_video()

image = read_images()


# plt.imshow(image)
# plt.title("Stock Image")
# plt.show()

# Press 'Q' to close window
def quit(subject):
    if subject.key == 'q':
        plt.close()


def image_grey(original_img):
    # Image conversion to black and white
    original_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

    # Show the image
    # plt.title("Image Grey Scaled")
    # plt.imshow(original_img, cmap='gray')
    # plt.show()
    return original_img


# Function to Blur the image
def image_blur(grey_image):
    # Bluring the image
    Img_blur = cv.GaussianBlur(grey_image, (5, 5), 0)

    # plt.title("Image Blurred")
    # plt.imshow(Img_blur, cmap='gray')
    # plt.show()
    return Img_blur


# The blurred Image returned (calls function)
# blur = image_blur(grey_Image)


# Canny Edge Detection - passing the blurred grey image
def image_canny(blur_img):
    img_canny = cv.Canny(blur_img, 100, 200)
    # plt.title("Canny Image")
    # plt.imshow(img_canny, cmap='gray')
    # plt.show()
    return img_canny


# Canny Image (calls function)
# canny = image_canny(blur)
# The grey Image returned (calls function)
grey_Image = image_grey(image)


# Function looks at just the white road lines.
# Using HSV Colourspace (Hue, Saturation, Value)
def identify_lines(img):
    original_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Identifying the white line using HSV placing into np.array
    upper_white = np.array([255, 7, 230], dtype=np.uint8)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)

    # Masking the white
    white_mask = cv.inRange(original_img, lower_white, upper_white)

    # A threshold to target only white colours in the image
    masked_image = cv.bitwise_and(img, white_mask)

    # plt.imshow(masked_image)
    # plt.title('Masked white image')
    # plt.show()

    return masked_image


lane_lines = identify_lines(grey_Image)


# Declaring a region of interest within an image
def region_of_interest(img):
    # Getting height and width of the image we want to use
    img_height = img.shape[0]
    img_width = img.shape[1]
    print("Image Height", img_height)
    print("Image Width", img_width, '\n')

    roi_triangle = np.array([
        [(0, 478), (440, 216), (490, 216), (img_width, img_height)]
    ])

    # Creating a mask for the image
    img_mask = np.zeros_like(img)

    # Combining the mask
    img_mask = cv.fillPoly(img_mask, roi_triangle, 255)
    img_mask = cv.bitwise_and(img, img_mask)

    # Displaying
    # plt.imshow(img_mask)
    # plt.title("Masked ROI Image")
    # plt.show()

    return img_mask


roi = region_of_interest(lane_lines)


def hough_algorithm():
    # This is an array
    hough_transform_output = cv.HoughLinesP(roi, rho=2, theta=3.642 / 180, threshold=100, minLineLength=40,
                                            maxLineGap=5)

    # Displaying
    # print(type(lane_lines_array))
    # print(lane_lines_array)

    return hough_transform_output


lane_lines_array = hough_algorithm()


def average_function(value):
    start = 0
    for i in value:
        start = start + i

    average = start / len(value)

    return average


def display(masked_image, output):
    black_img = np.zeros_like(masked_image, 'uint8')

    # Simple if statement to discover lines
    if output is not None:
        for single_line in output:
            x1, y1, x2, y2 = single_line

            print("x1 ", x1)
            print("y1 ", y1)
            print("x2 ", x2)
            print("y2 ", y2)

            # Drawing the line onto the black image
            cv.line(black_img, (x1, y1), (x2, y2), (0, 0, 255), 9)
    else:
        print("No lines detected")

    return black_img


def lane_line_average(declared_roi, hough_transform_lines):
    left_lane = []
    right_lane = []

    for line in hough_transform_lines:
        # Array from hough transformation
        print(line)
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
        print("y2_array_numpy", y2_array_numpy, "\n")

        # Polynomial fit (finds the least square polynominal fit)
        # (best fitting curve to a given set of points)
        # using x1, x2, y1, y2
        polyfit_value = np.polyfit((x1, x2), (y1, y2), 1)
        print("polyfit_value = ", polyfit_value)

        slope_of_line_seg = polyfit_value[0]
        y_intercept = polyfit_value[1]
        print("polyfit_value[0] (slope_of_line_seg) = ", polyfit_value[0])
        print("polyfit_value[1] (y_intercept) = ", polyfit_value[1], '\n')

        if slope_of_line_seg < 0:
            # If the slope is < 0 add to left_lines array
            left_lane.append((slope_of_line_seg, y_intercept))
        else:
            # else add to right_lines array
            right_lane.append((slope_of_line_seg, y_intercept))

    # We expect left line to start with NEGATIVE slope
    print("Left Line Values [NEGATIVE SLOPE] = ", left_lane)
    print("Right Line Values [POSITIVE SLOPE] = ", right_lane, '\n')

    # Averaging the values we got from left_lane, right_lane tuple arrays
    # Axis=0 computes the mean value over flattened array, along with rows
    left_lane_average = np.mean(left_lane, axis=0)
    right_lane_average = np.mean(right_lane, axis=0)
    print("Left lane average number: ", left_lane_average)
    print("Right lane average number: ", right_lane_average, "\n")

    # Adding the points to all masked image
    right_lane_line = calc_lane_point(roi, right_lane_average)
    left_lane_line = calc_lane_point(roi, left_lane_average)

    return [left_lane_line, right_lane_line]


# This function averages the lines taken in from the hough_algorithm() function
# It finds the average slope and y intercepts per line segment
# Displaying one slide line
def calc_lane_point(masked_image, lane_point_average):
    slope_of_line, y_intercept = lane_point_average

    # This is the image height of the given image we pass
    y1 = masked_image.shape[0]

    # This is how long we want the lines to be in our image
    y2 = int(y1 * (3 / 4))

    x1 = int((y1 - y_intercept) / slope_of_line)
    x2 = int((y2 - y_intercept) / slope_of_line)

    return [x1, y1, x2, y2]



# cap = cv.VideoCapture("video.avi")
# while (cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = image_canny(frame)
#     cropped_image = region_of_interest(canny_image)
#     lines = cv.HoughLinesP(cropped_image, rho=2, theta=3.642/180, threshold=100, minLineLength=35, maxLineGap=5) # (roi, rho=2, theta=3.642 / 180, threshold=100, minLineLength=40, maxLineGap=5)
#     averaged_lines = lane_line_average(frame, lines)
#     line_image = display(frame, averaged_lines)
#     combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv.imshow("result", combo_image)
#     cv.waitKey(1000)

'''## THE CORE OF THE WORK ##'''
#img_copy = cv.imread('images/Curve Road/image0006.png')
img_copy = np.copy(image)

# Step 1: Turn the image grey
grey = image_grey(img_copy)

# Step 2: Add the blur
blur_grey_img = image_blur(grey)

# Step 3: Canny Edge
edge_detection = image_canny(blur_grey_img)

# Step 4: Isolate our area of interest
region = region_of_interest(edge_detection)

# plt.imshow(region)
# plt.title("ROI")
# plt.show()

# Step 5: Hough Transformation
output = cv.HoughLinesP(region, rho=2, theta=3.642/180, threshold=100, minLineLength=35, maxLineGap=5)

# Step 6: Average the lines found
avg = lane_line_average(img_copy, output)

print("Average Lines", avg)

# Step 6: Add blue lines to mask
blue_lines = display(img_copy, avg)

# Step 7: Add the wights to the image
add_weights = cv.addWeighted(img_copy, 0.8, blue_lines, 1, 1)

# Step 8: Display
plt.imshow(add_weights)
plt.title("Please work")
plt.show()

read_images()
