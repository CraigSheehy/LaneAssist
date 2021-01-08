import cv2
import numpy as np
import matplotlib.pyplot as plt


def makeCords(image, line_parameters):
    slope, intercept = line_parameters
    print("Image Shape", image.shape)

    #y1 = image.shape[0] #1078 starting at the bottom
    y1 = 920
    y2 = int(y1 * (3 / 4)) #getting the middle
    #print(y2)

    # intercepts
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    #print(x2, "x2")
    #print(x1, "x1")
    return np.array([x1, y1, x2, y2]) #returning all cordinates as array


def averageSlope(image, lines):
    # co-ordinates for lines
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print("parameters", parameters)
        slope = parameters[0]
        intercept = parameters[1]

        # slope
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    left_average = np.average(left, axis=0)
    right_average = np.average(right, axis=0)
    print("right average", right_average)
    print("left average", left_average)

    left_line = makeCords(image, left_average)
    right_line = makeCords(image, right_average)

    return np.array([left_line, right_line])


def canny(image):
    # grey step 1
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur step 2
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # canny image step 3 (edge detection)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# payload
def displayLines(image, lines):
    # black mask
    line_image = np.zeros_like(image) # Black image, (0, 0, 0)
    #print(line_image)

    # display lines detected to black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            print("x1, y1", x1, y1)
            print("x2, y2", x2, y2)

            #drawing lines
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 9)
    return line_image


def regionOfInterest(image):
    height = image.shape[0]
    polyg = np.array([
        [(530, 916), (1550, 916), (1006, 580)]
    ])

    # black mask
    mask = np.zeros_like(image) #the white triangle

    # filling mask
    cv2.fillPoly(mask, polyg, 255)
    mask_image = cv2.bitwise_and(image, mask) #masking white on the canny

    # cv2.imshow("ROI", mask)
    return mask_image


# find the test image
# image = cv2.imread('images/newRef.png')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)

# cropping image
# cropped = regionOfInterest(canny_image)
# lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 86, np.array([]), minLineLength=40, maxLineGap=5)     #image(detect_lines), resolution of array(2px), degress of radates, threshold, placeholder_array, length of line in pixels, max distance for lines
#
# average_lines = averageSlope(lane_image, lines)

# Lined Image
# line_image = displayLines(lane_image, average_lines)

# combination image
# comboImage = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) #blending


# show image
# cv2.imshow("cropped", cropped)
# cv2.imshow("Normal Image", image) #stock image
# cv2.imshow("canny", canny_image)  #canny image
# cv2.imshow("ROI", regionOfInterest(canny_image)) #ROI discovered
# cv2.imshow("cropped", cropped)
# cv2.imshow("lined image", line_image)   #Lined Image

# plt.imshow(image)
# plt.show()
#
# cv2.imshow("Result", comboImage)
# cv2.waitKey(0)


cap = cv2.VideoCapture("images/sample.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = regionOfInterest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 70, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = averageSlope(frame,lines)
    line_image = displayLines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    cv2.waitKey(1)


# triangle
# bottom left
# 563, 918

# bottom right
# 1545, 917

# mid
# 1025, 600
