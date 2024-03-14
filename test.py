import numpy as np
import cv2

img1 = cv2.imread('./data/0000.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/0001.JPG', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
keypoint2, descriptor2 = sift.detectAndCompute(img1, None)

# cv2.imshow('d', img)
# cv2.waitKey()
# cv2.destroyAllWindows()