import datetime
from ultralytics import YOLO
import cv2
import numpy as np

def find_box_center(image,vis_masks = False,vis_output=False):
    # Define the region of interest (ROI) coordinates
    x = 280  # starting x-coordinate
    y = 80  # starting y-coordinate
    width = 180  # width of the ROI
    height = 320  # height of the ROI

    # Crop the image using numpy array slicing
    image = image[y:y+height, x:x+width]
    if vis_masks:
        cv2.imshow('image', image)


    # hsv filter
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_min, h_max = 0 , 5
    s_min, s_max = 0 , 5
    v_min, v_max = 90,  180

    lower_threshold = np.array([h_min, s_min, v_min])
    upper_threshold = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
    if vis_masks:
        cv2.imshow('mask', mask)

    result = cv2.bitwise_and(image, image, mask=mask)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(result, (5, 5), 0)
    if vis_masks:
        cv2.imshow('blurred', blurred)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    if vis_masks:
        cv2.imshow('edge', edges)


    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size according to your needs

    # Perform dilation
    dilated = cv2.dilate(edges, kernel, iterations=5)


    # Find contours in the mask
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the center of mask
    # Initialize variables for centroid calculation
    moments = cv2.moments(contours[0])
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])

    # Draw the center on the mask
    radius = 10
    cv2.circle(image, (center_x, center_y), radius, (0, 255, 255), -1)
    cv2.circle(image, (center_x, center_y), radius-2, (255, 0, 255), -1)
    if vis_output:
        cv2.imshow('image', image)
    if vis_masks or vis_output:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [center_x, center_y]




image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230704-151403.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-090410.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-094624.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-095040.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101956.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101841.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101714.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-101610.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110838.png"
# image_path = "sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110838.png"

image = cv2.imread(image_path)

box_pos = find_box_center(image,vis_masks=True,vis_output=True)
print("box_center:",box_pos)


# # Define the region of interest (ROI) coordinates
# x = 280  # starting x-coordinate
# y = 80  # starting y-coordinate
# width = 180  # width of the ROI
# height = 320  # height of the ROI

# # Crop the image using numpy array slicing
# image = image[y:y+height, x:x+width]
# cv2.imshow('image', image)


# # hsv filter
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# h_min, h_max = 0 , 5
# s_min, s_max = 0 , 5
# v_min, v_max = 90,  180

# lower_threshold = np.array([h_min, s_min, v_min])
# upper_threshold = np.array([h_max, s_max, v_max])
# mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
# cv2.imshow('mask', mask)

# result = cv2.bitwise_and(image, image, mask=mask)

# # Gaussian blur to reduce noise
# blurred = cv2.GaussianBlur(result, (5, 5), 0)
# cv2.imshow('blurred', blurred)

# # Apply Canny edge detection
# edges = cv2.Canny(blurred, 100, 200)
# cv2.imshow('edge', edges)


# # Define the kernel for dilation
# kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size according to your needs

# # Perform dilation
# dilated = cv2.dilate(edges, kernel, iterations=5)


# # Find contours in the mask
# contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # find the center of mask
# # Initialize variables for centroid calculation
# moments = cv2.moments(contours[0])
# center_x = int(moments['m10'] / moments['m00'])
# center_y = int(moments['m01'] / moments['m00'])

# # Draw the center on the mask
# radius = 10
# cv2.circle(image, (center_x, center_y), radius, (0, 255, 255), -1)
# cv2.circle(image, (center_x, center_y), radius-2, (255, 0, 255), -1)



# # Perform erosion
# eroded = cv2.erode(dilated, kernel, iterations=3)

# # Display the original, eroded, and dilated images
# # cv2.imshow('Original', image)
# cv2.imshow('Eroded', eroded)
# cv2.imshow('Dilated', dilated)



# # # Find contours in the edge map
# # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # Initialize variables to store the largest rectangle
# # max_area = 0
# # max_contour = None

# # # Iterate over the contours
# # for contour in contours:
# #     # Approximate the contour as a polygon
# #     epsilon = 0.02 * cv2.arcLength(contour, True)
# #     approx = cv2.approxPolyDP(contour, epsilon, True)

# #     # If the contour has four corners, it is likely a rectangle
# #     if len(approx) == 4:
# #         # Calculate the area of the contour
# #         area = cv2.contourArea(approx)

# #         # Update the maximum area and contour if necessary
# #         if area > max_area:
# #             max_area = area
# #             max_contour = approx

# # # Draw the largest rectangle on the image
# # if max_contour is not None:
# #     cv2.drawContours(image, [max_contour], 0, (0, 255, 0), 2)



# cv2.imshow('Result', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


############################################################################
# define some constants
# CONFIDENCE_THRESHOLD = 0.05
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)

# # load the pre-trained YOLOv8n model
# model = YOLO("yolov8n.pt")

# image_path = 'sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110939.png'
# frame = cv2.imread(image_path)

# # Define the region of interest (ROI) coordinates
# x = 320  # starting x-coordinate
# y = 80  # starting y-coordinate
# width = 150  # width of the ROI
# height = 280  # height of the ROI

# # Crop the image using numpy array slicing
# cropped_image = frame[y:y+height, x:x+width]

# detections = model(cropped_image)[0]
# print (detections)

# # loop over the detections
# for data in detections.boxes.data.tolist():
#     # extract the confidence (i.e., probability) associated with the detection
#     confidence = data[4]

#     # filter out weak detections by ensuring the 
#     # confidence is greater than the minimum confidence
#     if float(confidence) < CONFIDENCE_THRESHOLD:
#         continue

#     # if the confidence is greater than the minimum confidence,
#     # draw the bounding box on the frame
#     xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
#     cv2.rectangle(cropped_image, (xmin, ymin) , (xmax, ymax), GREEN, 2)

# cv2.imshow('frame',cropped_image)
# cv2.waitKey(0)


#################################################################################################
# import cv2
# import numpy as np

# def detect_box(image):

#     # Define the region of interest (ROI) coordinates
#     x = 0  # starting x-coordinate
#     y = 0  # starting y-coordinate
#     width = 640  # width of the ROI
#     height = 480  # height of the ROI

#     # Crop the image using numpy array slicing
#     cropped_image = image[y:(y+height), x:(x+width)]
#     cv2.imshow('cropped', cropped_image)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # cv2.imshow('blurred', blurred)

#     # Apply Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)
#     cv2.imshow('edge', edges)
#     # Find contours in the image
#     contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Iterate over the contours and find the rectangle
#     for contour in contours:
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

#         # If the approximated contour has four points, it's likely a rectangle
#         if len(approx) == 4:
#             return approx

#     return None

# # Load the image
# image_path = 'sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest20230705-110939.png'
# image = cv2.imread(image_path)

# # Detect the rectangle in the image
# rectangle = detect_box(image)

# if rectangle is not None:
#     # Draw the detected rectangle on the image
#     cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 3)
#     cv2.imshow('Detected Rectangle', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print('No rectangle detected in the image.')