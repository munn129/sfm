import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
def load_image(image_path) -> list:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise Exception(f'read {image_path} is failed')
    
    return img

def load_K(K_path) -> list:
    K = []
    with open(K_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            K.append(row)        
    return np.array(K)

def sift_des_generator(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe's ratio test
    matches_good = [m1 for m1, m2 in matches if m1.distance < 0.75*m2.distance]

    return matches_good

def main():
    pass

if __name__ == '__main__':
    main()