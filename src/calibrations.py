# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

def defineReference(frame, t_circle=3, color_circle = [0,0,0]):
    global coord
    coord = [0,0]
    #this function will be called whenever the mouse is right-clicked
    def mouse_callback(event, x, y, flags, params):
        global coord
        #right-click event value is 2
        if event == cv2.EVENT_LBUTTONDOWN:
            #store the coordinates of the right-click event
            coord = [x, y]
            #this just verifies that the mouse data is being collected
            #you probably want to remove this later
    img_point = np.copy(frame)
    cv2.imshow("Image", img_point)
    cv2.setMouseCallback('Image', mouse_callback)
    while True:
        img_point = np.copy(frame)
        cv2.circle(img_point, (coord[0], coord[1]), t_circle, color_circle, -1)
        cv2.imshow("Image", img_point)

        key = cv2.waitKey(25)
        if key != -1:
            print(coord)
            cv2.destroyAllWindows()
        if(key == 13 or key==141): #enter
            print("Saving reference")
            break
    return coord

def getMmppInterface(frame, object_x_mm=7.5, object_y_mm=7.5):
    frame_tmp = np.copy(frame)
    # Select ROI
    r = cv2.selectROI(frame_tmp)
    cv2.destroyAllWindows()
    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    side = True
    while True:
        # Crop image
        img_crop = frame_tmp[y:y+h, x:x+w]
        cv2.imshow("Image", img_crop)
        # Display cropped image
        key = cv2.waitKey(25)
        if key != -1:
            cv2.destroyAllWindows()
        if key == ord('q'):
            print("Not saving...") 
            break
        elif key == 97: #left
            if side:
                x -= 1
                w += 1
            else:
                w -= 1
        elif key == 119: #up
            if side:
                y -= 1
                h += 1
            else:
                h -= 1
        elif key == 100: #right
            if side:
                x += 1
                w -= 1
            else:
                w += 1
        elif key == 115: #down
            if side:
                y += 1
                h -= 1
            else:
                h += 1
        elif (key == 32): #space
            side = not(side)
        elif(key == 13 or key==141): #enter
            print("Saving conversion")
            break
    p_x = w
    p_y = h

    mmppx = object_x_mm/p_x
    mmppy = object_y_mm/p_y
    return mmppx, mmppy