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
global_descriptors = []

for image in tqdm(images[:10], desc='sift detect and compute'):
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
    match = [m1 for m1, m2 in _match if m1.distance < 0.99 * m2.distance]
    matches.append(match)

# _match = bf.knnMatch(descriptors[0], descriptors[1], k=2)
# match = [m1 for m1, m2 in _match if m1.distance < 0.8 * m2.distance]

print('knn match and Lowe\'s ratio test are done' )


# matches optimization with epipolar constraint
# matches_optmz = [[query_pixel, train_pixel], ... ]
matches_optmz = []
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

########################################

# query_idx = [i.queryIdx for i in match]
# train_idx = [i.trainIdx for i in match]

# # 매칭된 키포인트의 픽셀들을 저장함
# query_pixel = np.float32([keypoints[0][i].pt for i in query_idx])
# train_pixel = np.float32([keypoints[1][i].pt for i in train_idx])
# train_des = np.array([descriptors[1][i] for i in train_idx])

# E, mask = cv2.findEssentialMat(query_pixel, train_pixel, K, method=cv2.RANSAC)
# # 쿼리와 트레인의 매칭 픽셀 수는 같고(당연한 이야기지만...)
# # 이 두 점을 삼각측량해서 3차원 포인트로 만듦
# query_pixel = query_pixel[mask.ravel() == 1]
# train_pixel = train_pixel[mask.ravel() == 1]
# train_des = train_des[mask.ravel() == 1]

# # 나중에 코드 수정해야 됨
# pixel_1 = query_pixel
# pixel_2 = train_pixel

# add color
for p in pixel_1:
    colors.append(images[0][int(p[0])][int(p[1])])

################################

# decompose essential matrix -> camera extrinsic
# 첫 번째 이미지와 두 번째 이미지 사이의 extrinsic
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

structure = []
for p1, p2 in zip(h_pixel_1, h_pixel_2):
    point = triangulation(Rt0, Rt1, p1, p2)
    structure.append(point)

##### growing step #####
# global_descriptors = np.concatenate((descriptors[0], descriptors[1]))

# _match = bf.knnMatch(descriptors[1], descriptors[2], k=2)
# match = [m1 for m1, m2 in _match if m1.distance < 0.8 * m2.distance]

# query_idx = [i.queryIdx for i in match]
# train_idx = [i.trainIdx for i in match]
    
# query_pixel = np.float32([keypoints[1][i].pt for i in query_idx])
# train_pixel = np.float32([keypoints[2][i].pt for i in train_idx])

# 1. 두 번째 이미지랑 세 번째 이미지랑 매칭
# 2. 매칭 된 포인트 중 두 번째 이미지에서 첫 번째 이미지랑 매칭된 포인트 정리
# 즉, 1~2 사이에서도 매칭되고, 2~3 사이에서도 매칭 된 애들을 정리해야 됨
# 그리고 1~2 사이에서 매칭된 3차원 포인트와 2~3 사이에서 매칭된 2차원 포인트(세 번째 이미지)
# 이 둘을 갖고 solvepnp 돌려서 세 번째 이미지의 위치를 구하고 그 다음 삼각 측량
# 어떻게?

# 1. 두번째 이미지와 세 번째 이미지와 매칭 결과 matches_optmz[1]
# 두 번째 이미지의 매칭 포인트 matches_optmz[1][0]
# 세 번째 이미지의 매칭 포인트 matches_optmz[1][1]
# 근데 여기서 1~2 사이에서 매칭된 두 번째 이미지의 포인트 matches_optmz[0][1]
# 그럼 matches_optmz[0][1]이랑 [1][0] 이랑 같은 걸 찾아야 한다.
# 1~2 사이는 mask로 필요하고(3차원 포인트에서 뽑아야 됨)
# 2~3 사이는 그 점들만 있으면 된다.

g_mask = []
points_2to3 =[]

for i in matches_optmz[0][1]:
    if i in matches_optmz[1][0]:
        g_mask.append(1)
        points_2to3.append(i)
    else:
        g_mask.append(0)

# 1~2단계에서도 매칭되고, 2~3 단계에서도 매칭된 3차원 포인트
structure_for_matching = np.array([structure[i] for i, val in enumerate(g_mask) if val == 1])
points_2to3 =- np.array(points_2to3)

retval, r, t, inliers = cv2.solvePnPRansac(structure_for_matching, points_2to3, K, None)
R, _ = cv2.Rodrigues(r)
h_Rt_mat = np.hstack((R, t))
h_Rt_mat = K @ h_Rt_mat
h_Rt_mat = np.vstack((h_Rt_mat, [0, 0, 0, 1]))

# 이제 여기서 2~3 매칭된 점을 삼각 측량
h_pixel_1 = []
h_pixel_2 = []

for p1, p2 in zip(matches_optmz[1][0], matches_optmz[1][1]):
    h_pixel_1.append(np.append(p1.flatten(), 1))
    h_pixel_2.append(np.append(p2.flatten(), 1))

h_pixel_1 = np.array(h_pixel_1)
h_pixel_2 = np.array(h_pixel_2)

for p1, p2 in zip(h_pixel_1, h_pixel_2):
    point = triangulation(Rt1, h_Rt_mat, p1, p2)
    structure.append(point)
    
#################################################


# for idx in tqdm(range(len(matches_optmz) - 1), desc = 'growing step'):
#     g_mask = []
#     points_2to3 = []

#     # 직전 매칭의 1번째 이미지와 현재 매칭의 0번째 매칭
#     for i in matches_optmz[idx][1]:
#         if i in matches_optmz[idx + 1][0]:
#             g_mask.append(1)
#             points_2to3.append(i)
#         else:
#             g_mask.append(0)

#     # 1~2단계에서도 매칭되고, 2~3 단계에서도 매칭된 3차원 포인트
#     structure_for_matching = np.array([structure[i] for i, val in enumerate(g_mask) if val == 1])
#     points_2to3 =- np.array(points_2to3)

#     _, r, t, _ = cv2.solvePnPRansac(structure_for_matching, points_2to3, K, None)
#     R, _ = cv2.Rodrigues(r)
#     h_Rt_mat = np.hstack((R, t))
#     h_Rt_mat = K @ h_Rt_mat
#     h_Rt_mat = np.vstack((h_Rt_mat, [0, 0, 0, 1]))

#     # 이제 여기서 2~3 매칭된 점을 삼각 측량
#     h_pixel_1 = []
#     h_pixel_2 = []

#     for p1, p2 in zip(matches_optmz[idx+ 1 ][0], matches_optmz[idx + 1][1]):
#         h_pixel_1.append(np.append(p1.flatten(), 1))
#         h_pixel_2.append(np.append(p2.flatten(), 1))

#     h_pixel_1 = np.array(h_pixel_1)
#     h_pixel_2 = np.array(h_pixel_2)

#     for p1, p2 in zip(h_pixel_1, h_pixel_2):
#         point = triangulation(Rt1, h_Rt_mat, p1, p2)
#         structure.append(point)


#################################################################

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

# visualize_3d(np.array(structure).T)

def pts2ply(pts):
    f = open('101010101010101010.ply','w')
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
    
    for pt in pts: 
        f.write('{} {} {} 100 100 100\n'.format(pt[0],pt[1],pt[2]))
        # f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],c[0],c[1],c[2]))
    f.close()

# tmp = []
# for i in structure:
#     if i[0] < 0: tmp.append(i)

pts2ply(np.array(structure))