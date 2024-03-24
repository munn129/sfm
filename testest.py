'''
지금 고민인거
growing step을 어떻게 해야할 지 모르겠음.

1번째 2번째는 ㅇㅋ
근데 3번째는?

- 일단 2d3d matching하고(1번째~2번째 매칭의 2번째 디스크립터 <-> 2번째~3번째 매칭의 2번째 디스크립터)
- 근데 생각해보니까 위에거 아닌듯; 
- 일단 solvepnp를 쓰려면 초기 3차원 점들이랑 3번째 이미지가 매칭이 되어야 함
- 3차원 좌표랑 그 3차원 좌표에 이미지에 투영된 2차원 좌표를 알고 있어야 함
- 그럼 디스크립터는 필요 없고...
- 그냥 1~2 매칭의 2 이미지 포인트(3차원)와 2~3 매칭의 2이미지 포인트를 알아야 함
'''

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

# matches optimization with epipolar constraint
# matches_optmz = [[query_pixel, train_pixel], ... ]
matches_optmz = []
# match된 descriptor의 배열, 즉 3차원 포인트가 될 2차원 디스크립터 [i][1]와 새로운 이미지의 디스크립터[i+1][0] 
# 이 둘의 매칭으로 3d2d matching 구현
# matches_des = []
for id, match in enumerate(matches):
    query_idx = [i.queryIdx for i in match]
    train_idx = [i.trainIdx for i in match]
    
    query_pixel = np.float32([keypoints[id][i].pt for i in query_idx])
    train_pixel = np.float32([keypoints[id + 1][i].pt for i in train_idx])

    # query_des = np.array([descriptors[id][i] for i in query_idx])
    # train_des = np.array([descriptors[id + 1][i] for i in train_idx])

    _, mask = cv2.findEssentialMat(query_pixel, train_pixel, K, method=cv2.RANSAC)
    matches_optmz.append((query_pixel[mask.ravel() == 1], train_pixel[mask.ravel() == 1]))
    # matches_des.append((query_des[mask.ravel() == 1], train_des[mask.ravel() == 1]))

# Initialization step
pixel_1 = matches_optmz[0][0]
pixel_2 = matches_optmz[0][1]
E, _ = cv2.findEssentialMat(pixel_1, pixel_2, K, method=cv2.RANSAC)

for p in pixel_1:
    colors.append(images[0][int(p[0])][int(p[1])])

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
    tmp = np.array([
        p1[1]*Rt0[2,:] - Rt0[1,:],
        -(p1[0]*Rt0[2,:] - Rt0[0,:]),
        p2[1]*Rt1[2,:] - Rt1[1,:],
        -(p2[0]*Rt1[2,:] - Rt1[0,:])
    ]).reshape((4, 4))
    
    _, _, Vt = np.linalg.svd(tmp.T @ tmp)
    
    return Vt[3, 0:3]/Vt[3,3]

init_structure = []
for p1, p2 in zip(h_pixel_1, h_pixel_2):
    point = triangulation(Rt0, Rt1, p1, p2)
    init_structure.append(point)

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

# 둘이 같은 이미지에서 나왔지만 서로 다른 이미지와 매칭된 특징점들임
mask_prev = np.zeros(len(matches_optmz[0][1]))
mask_pres = np.zeros(len(matches_optmz[1][0]))

# 1~2사이에 매칭된 이미지2의 특징점이 2~3사이에 매칭된 이미지2의 특징점이랑 같을 때
tmp_3d = []
tmp_2d = []

for i, point in enumerate(matches_optmz[0][1]):
    if np.any(np.all(np.isclose(matches_optmz[1][0], point), axis=1)):
        mask_prev[i] = 1
        tmp_3d.append(point)

for i, point in enumerate(matches_optmz[1][0]):
    if np.any(np.all(np.isclose(matches_optmz[0][1], point), axis=1)):
        mask_pres[i] = 1
        tmp_2d.append(point)

# visualize_3d(np.array(init_structure).T)

def pts2ply(pts, color):
    f = open('result.ply','w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(pts.shape[0]))
    
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    
    f.write('end_header\n')
    
    for pt, c in zip(pts, color): 
        # f.write('{} {} {} 255 255 255\n'.format(pt[0],pt[1],pt[2]))
        f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],c[0],c[1],c[2]))
    f.close()

pts2ply(np.array(init_structure), colors)