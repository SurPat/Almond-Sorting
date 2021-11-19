from __future__ import print_function
import cv2
import numpy as np
import imutils 

flag_detected = 0
Red_Counters = 0

def rescale_frame(frame, percent=80):  # make the video windows a bit smaller
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


img = cv2.imread("ten.jpg")
frame = rescale_frame(img)
out_new = np.uint8(frame)
out_Gray = cv2.cvtColor(out_new, cv2.COLOR_BGR2GRAY)
ret,thresh_out = cv2.threshold(out_Gray,127,255,cv2.THRESH_BINARY_INV)
kernel_ip = np.ones((2,2),np.uint8)
eroded_ip = cv2.erode(thresh_out,kernel_ip,iterations = 1)
dilated_ip = cv2.dilate(eroded_ip,kernel_ip,iterations = 1)
cnts = cv2.findContours(dilated_ip.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


if len(cnts) == 0:
    flag_empty = 1
    flag_detected = 0
    cv2.imshow("output", frame)
    cv2.waitKey(30)
    
# converting  BGR to HSV Frame
Big_faulty = max(cnts, key = cv2.contourArea)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# the range of soaked almond
blu_lower = np.array([0, 61,190], np.uint8)
blu_upper = np.array([102, 255, 255], np.uint8)

#the range of dry almond
dry_lower = np.array([0, 206,229], np.uint8)
dry_upper= np.array([255, 255, 255], np.uint8)

# finding the range of dry and soaked in the image
red = cv2.inRange(hsv, blu_lower, blu_upper)
dry = cv2.inRange(hsv, dry_lower, dry_upper)

kernal = np.ones((3, 3), "uint8")

# dilation of the image ( to remove noise) create mask for red color
red = cv2.dilate(red, kernal, iterations=1)
res = cv2.bitwise_and(frame, frame, mask=red)
cv2.imshow("soaked",red)
contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# dilation of the image ( to remove noise) create mask for dry almond
dry = cv2.dilate(dry, kernal, iterations=1)
dry1 = cv2.bitwise_and(frame, frame, mask=dry)
cv2.imshow("Dry",dry)
contours1, hierarchy = cv2.findContours(dry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    c = cv2.contourArea(contour)
    if (c >=2000):
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, "soaked", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        
for contour in contours1:
    x, y, w, h = cv2.boundingRect(contour)
    c = cv2.contourArea(contour)
    if (c >=1500):
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(frame, "dry", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)

cv2.imshow("output", frame)
cv2.waitKey(0)



