import numpy as np
import cv2

from calibrated_fivepoint import calibrated_fivepoint

filename = './data/K.txt'
K = []
with open(filename, 'r') as file:
    for line in file:
        row = list(map(float, line.strip().split()))
        K.append(row)        
K = np.array(K)


img1 = cv2.imread('./data/0000.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/0001.JPG', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_parmas = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_parmas)
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

# Lowe's ratio test
for i, (m,n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
E, mask = cv2.findEssentialMat(pts1, pts2, K)
EE = calibrated_fivepoint(pts1, pts2)

r1, r2, t = cv2.decomposeEssentialMat(E)
retval, rot, tran, m = cv2.recoverPose(E, pts1, pts2)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
 
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None)
# cv2.imshow('d', img3)
# cv2.waitKey()
# cv2.destroyAllWindows()