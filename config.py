import os
import numpy as np

image_dir = './data/'
MRT = 0.7
#相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距，而K[0][2]和K[1][2]
#代表图像的中心像素。
K = np.array([
        [1698.873755, 0, 971.7497705],
        [0, 1698.8796645,  647.7488275],
        [0, 0, 1]])

#选择性删除所选点的范围。
x = 0.5
y = 1