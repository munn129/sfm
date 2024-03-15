import numpy as np
import cv2

def calibrated_fivepoint(Q1, Q2):
    # reshape Q1 and Q2
    Q1 = Q1.T
    Q2 = Q2.T

    # For linear from of Essential Matrix
    Q = np.column_stack((Q1[:,0] * Q2[:,0],
                   Q1[:,1] * Q2[:,0],
                   Q1[:,2] * Q2[:,0],
                   Q1[:,0] * Q2[:,1],
                   Q1[:,1] * Q2[:,1],
                   Q1[:,2] * Q2[:,1],
                   Q1[:,0] * Q2[:,2],
                   Q1[:,1] * Q2[:,2],
                   Q1[:,2] * Q2[:,2]))
    
    U, S, V = np.linalg.svd(Q, full_matrices=False)
    EE = V[-4:,5:8]

    return V

def calibrated_fivepoint_helper(EE):
    pass