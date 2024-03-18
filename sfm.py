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
Rt1 = K @ Rt1

def homogeneous_point(pts1, pts2, length):
    h_pt1 = [[]]
    h_pt2 = [[]]
    
    for i in range(length):
        tmp1 = pts1[i].flatten()
        tmp1 = np.append(tmp1, 1)
        h_pt1 = np.append(h_pt1, tmp1)
        tmp2 = pts2[i].flatten()
        tmp2 = np.append(tmp2, 1)
        h_pt2 = np.append(h_pt2, tmp2)

    h_pt1 = h_pt1.reshape((length), 3)
    h_pt2 = h_pt2.reshape((length), 3)

    return h_pt1, h_pt2

pts1, pts2 = homogeneous_point(pts1, pts2, len(pts1))

def linear_triangulation(Rt0, Rt1, pts1, pts2):
    A = [pts1[1]*Rt0[2,:] - Rt0[1,:],
         -(pts1[0]*Rt0[2,:] - Rt0[0,:]),
         pts2[1]*Rt1[2,:] - Rt1[1,:],
         -(pts2[0]*Rt1[2,:] - Rt1[0,:])]
    
    A = np.array(A).reshape((4,4))
    AA = A.T @ A
    U, S, Vt = np.linalg.svd(AA)

    return Vt[3,0:3]/Vt[3,3]

p3ds = []
for pt1, pt2 in zip(pts1, pts2):
    p3d = linear_triangulation(Rt0, Rt1, pts1, pts2)
    p3ds.append(p3d)
p3ds = np.array(p3ds).T

X = np.array([])
Y = np.array([])
Z = np.array([])

X = np.concatenate((X, p3ds[0]))
Y = np.concatenate((Y, p3ds[1]))
Z = np.concatenate((Z, p3ds[2]))