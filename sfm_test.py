import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
def load_image(image_path) -> list:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise Exception(f'reading {image_path} is failed')
    
    return img

def load_K(K_path) -> list:
    K = []
    with open(K_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            K.append(row)        
    return np.array(K)

def sift_matcher(img1, img2) -> tuple:
    sift = cv2.SIFT_create()

    query_kp, query_des = sift.detectAndCompute(img1, None)
    train_kp, train_des = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_des, train_des, k=2)
    # Lowe's ratio test
    matches_good = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]
    query_idx = [i.queryIdx for i in matches_good]
    train_idx = [i.trainIdx for i in matches_good]
    p1 = np.float32([query_kp[i].pt for i in query_idx])
    p2 = np.float32([train_kp[i].pt for i in train_idx])

    return p1, p2

def find_E_with_RANSAC(p1, p2):
    # p1, p2 -> call by reference
    E, mask = cv2.findEssentialMat(p1, p2, method=cv2.RANSAC)
    p1 = p1[mask.ravel() == 1]
    p2 = p2[mask.ravel() == 1]
    
    return E

def main():
    pass

if __name__ == '__main__':
    main()