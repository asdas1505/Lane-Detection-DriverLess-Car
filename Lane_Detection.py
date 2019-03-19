import cv2
import numpy as np
import math

img = cv2.imread('lane3.jpg',1)

# Image is converted in Gray Scale.

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Now we will make a mask for our image which will specify the colors of interest in our image.

mask_white = cv2.inRange(gray_img, 200, 255)

gauss_img = cv2.GaussianBlur(mask_white, (5,5), 0)

# Now we will find the edges in the image by canny edge algorithm
# We define low threshold and high threshold
# Recommended ratio for high and low threshold value should be 1:3 or 1:2

low_threshold = 100
high_threshold = 200

canny_edge_img = cv2.Canny( gauss_img, low_threshold, high_threshold)

# Now we are going to mark our ROI i.e. Region of interest
# First we create a mask of similar size of our image

blank_mask = np.zeros_like(canny_edge_img)

# Then we assign colour to our region according to number of channels in our image

if len(canny_edge_img.shape) > 2:
    no_of_channels = canny_edge_img.shape[2]
    color_blank = (255,)*no_of_channels

else:
    color_blank = 255
    
# Now we will define the dimensions of box which is our region of interest
# Arrange vertices in Clockwise direction
img_shape = canny_edge_img.shape
upper_right = [img_shape[1] - img_shape[1]/2, img_shape[0]/1.74]
lower_right = [ img_shape[1] - img_shape[1]/200 , img_shape[0]]
lower_left = [0 , img_shape[0] ] 
upper_left = [img_shape[1]/2.1 , img_shape[0]/1.64]

vertices = [ np.array([lower_left, upper_left, upper_right, lower_right], dtype=np.int32)]

# We make the ROI of white color and apply it on our blank mask

cv2.fillPoly(blank_mask, vertices, color_blank)

# Now we will apply our blank mask on our image to get our ROI

ROI_img = cv2.bitwise_and(canny_edge_img, blank_mask)

# Now we will apply Hough Transform to make lines on our edges in our image

#threshold = 20
#min_line_len = 50
#max_line_gap = 200
lines = cv2.HoughLinesP(
    ROI_img,
    rho=2,
    theta=np.pi / 180,
    threshold=10,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=160
)
line_img = np.zeros((ROI_img.shape[0], ROI_img.shape[1], 3), dtype=np.uint8)

# Now we will draw lines

print(lines)

for line in lines:
    print(line)
    for x1,y1,x2,y2 in line:
        cv2.line(line_img,(x1,y1),(x2,y2), [0,255,0], 2)

new_img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)


cv2.imshow('image',new_img)
cv2.waitkey(0)
cv2.destroyAllWindows()







