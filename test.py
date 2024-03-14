import numpy as np
import cv2

img = cv2.imread('./Data/0000.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow(gray)