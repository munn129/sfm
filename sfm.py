import numpy as np
import cv2
import matplotlib.pyplot as plt

# from calibrated_fivepoint import calibrated_fivepoint

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

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# r1, r2, t = cv2.decomposeEssentialMat(E)
retval, R, t, m = cv2.recoverPose(E, pts1, pts2)

Rt0 = np.hstack((np.eye(3), np.zeros((3,1))))
Rt1 = np.hstack((R,t))
Rt1 = np.matmul(K, Rt1)

tri_pt1 = np.transpose(pts1)
tri_pt2 = np.transpose(pts2)

p3ds = []

for pt1, pt2 in zip(tri_pt1, tri_pt2):
    p3d = cv2.triangulatePoints(Rt0, Rt1, tri_pt1, tri_pt2)
    p3d /= p3d[3]
    p3ds.append(p3d)

p3ds = np.array(p3ds).T

X = np.array([])
Y = np.array([])
Z = np.array([])

X = np.concatenate((X, p3ds[0]))
Y = np.concatenate((Y, p3ds[1]))
Z = np.concatenate((Z, p3ds[2]))

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter(X, Y, Z, c='b', marker='o')
plt.show()