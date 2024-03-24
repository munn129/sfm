import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_root_dir = './data/'
images_dir = [ f'{dataset_root_dir}{i:04d}.JPG' for i in range(0,32)]
images = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in images_dir]

K = []
with open('./data/K.txt', 'r') as file:
    for line in file:
        row = list(map(float, line.strip().split()))
        K.append(row)
K = np.array(K)

# sift detect
sift = cv2.xfeatures2d.SIFT_create()

keypoints = []
descriptors = []
colors = []

for image in tqdm(images, desc='sift detect and compute'):
    kp, des = sift.detectAndCompute(image, None)
    
    keypoints.append(kp)
    descriptors.append(des)

print('keypoints and descriptors are extracted')

# knn match and lowe's SIFT ratio test
bf = cv2.BFMatcher()

# matches[0] -> between descriptor[0] and descriptor[1]
# len(matches) == len(descriptor) - 1
matches = []
for idx, val in enumerate(descriptors):
    if idx == (len(descriptors) - 1): break
    _match = bf.knnMatch(descriptors[idx], descriptors[idx + 1], k = 2)
    match = [m1 for m1, m2 in _match if m1.distance < 0.8 * m2.distance]
    matches.append(match)

print('knn match and Lowe\'s ratio test are done' )

# Initialization step
query_idx = [match.queryIdx for match in matches[0]]
train_idx = [match.trainIdx for match in matches[0]]

pixel_1 = np.float32([keypoints[0][i].pt for i in query_idx])
pixel_2 = np.float32([keypoints[1][i].pt for i in train_idx])

E, mask = cv2.findEssentialMat(pixel_1, pixel_2, K, method=cv2.RANSAC)
pixel_1 = pixel_1[mask.ravel() == 1]
pixel_2 = pixel_2[mask.ravel() == 1]

# decompose essential matrix -> camera extrinsic
camera_extrinsic = np.array([[]])
U, _, Vt = np.linalg.svd(E, full_matrices=True)
W = np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])
_camera_extrinsic = np.array([np.column_stack((U @ W @ Vt, U[:,2])),
                              np.column_stack((U @ W @ Vt, -U[:,2])),
                              np.column_stack((U @ W.T @ Vt, U[:,2])),
                              np.column_stack((U @ W.T @ Vt, -U[:,2]))])

for cm in _camera_extrinsic:
    for i in range(len(pixel_1)):
        p1 = pixel_1[i].flatten()
        p2 = pixel_2[i].flatten()
        p1p2 = np.concatenate((p1, p2))
        
        if np.any(cm @ p1p2.T < 0): break
        else: camera_extrinsic = cm

# for homogeneous coordinate
h_pixel_1 = []
h_pixel_2 = []

for p1, p2 in zip(pixel_1, pixel_2):
    h_pixel_1.append(np.append(p1.flatten(), 1))
    h_pixel_2.append(np.append(p2.flatten(), 1))

h_pixel_1 = np.array(h_pixel_1)
h_pixel_2 = np.array(h_pixel_2)

Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
Rt1 = K @ camera_extrinsic

# triangulation
def triangulation(Rt0, Rt1, p1, p2):
    A = np.array([
        p1[1]*Rt0[2,:] - Rt0[1,:],
        -(p1[0]*Rt0[2,:] - Rt0[0,:]),
        p2[1]*Rt1[2,:] - Rt1[1,:],
        -(p2[0]*Rt1[2,:] - Rt1[0,:])
    ]).reshape((4, 4))
    
    _, _, Vt = np.linalg.svd(A.T @ A)
    
    return Vt[3, 0:3]/Vt[3,3]

points = []
for p1, p2 in zip(h_pixel_1, h_pixel_2):
    point = triangulation(Rt0, Rt1, p1, p2)
    points.append(point)

points = np.array(points).T

def visualize_3d(p3ds):
    X = np.array([])
    Y = np.array([])
    Z = np.array([]) #120 
    X = np.concatenate((X, p3ds[0]))
    Y = np.concatenate((Y, p3ds[1]))
    Z = np.concatenate((Z, p3ds[2]))

    fig = plt.figure(figsize=(15,15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c='b', marker='o') 
    plt.show()

visualize_3d(points)