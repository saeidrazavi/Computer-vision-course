import cv2
import numpy as np
import matplotlib.pyplot as plt


def norimilize_matrix(src, hom):
    final = np.copy(hom)
    h, w = src.shape[:2]
    p = [[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]]
    p_prime = np.array(np.dot(hom, p))
    p_zegond = p_prime/p_prime[2, :]
    x_min = np.min(p_zegond[0, :])
    y_min = np.min(p_zegond[1, :])
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if(x_min < 0):
        t[0, 2] = x_min*-1.4
    if(y_min < 0):
        t[1, 2] = y_min*-1.4
    return np.dot(t, final)
  # ------------------------------------------


# ---- read the images
logo = np.array(cv2.imread("logo.png"))
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

focal_length = 500
angle = -np.arctan(40/25)
hx, hy = logo.shape[:2]
px = hx/2
py = hy/2


K = np.array([[focal_length, 0, px, 0], [
             0, focal_length, py, 0], [0, 0, 1, 0]])
Kinv = np.zeros([4, 3])
Kinv[:3, :3] = np.linalg.inv(K[:3, :3])*focal_length
Kinv[-1, :] = [0, 0, 1]


K = np.array([[focal_length, 0, 1000, 0], [
             0, focal_length, 1000, 0], [0, 0, 1, 0]])

T = np.array([[1, 0, 0, 0],
              [0, 1, 0, 40],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

RX = np.array([[1,           0,            0, 0],
               [0, np.cos(angle), -np.sin(angle), 0],
               [0, np.sin(angle), np.cos(angle), 0],
               [0,           0,            0, 1]])

RY = np.array([[1, 0, 0, 0],
               [0, 1,            0, 0],
               [0, 0, 1, 0],
               [0, 0,            0, 1]])

RZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0,            0, 1, 0],
               [0,            0, 0, 1]])

# Composed rotation matrix with (RX,RY,RZ)
R = np.linalg.multi_dot([RZ, RY, RX])
RT_inverse = np.linalg.inv(np.dot(R, T))
homograpgy_matrix = np.linalg.multi_dot([K, RT_inverse, Kinv])
homograpgy_matrix /= homograpgy_matrix[2, 2]
print(f"homography matrix = {homograpgy_matrix}")

# normilze matrix
homo = norimilize_matrix(logo, homograpgy_matrix)
homo /= homo[2, 2]
dst = cv2.warpPerspective(logo, homo, (2000, 2000),
                          cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

# save the image
plt.imsave("res12.jpg", dst.astype(np.uint8))
