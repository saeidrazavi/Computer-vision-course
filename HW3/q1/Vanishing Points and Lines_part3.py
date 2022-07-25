from turtle import color, width
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
import os
import moviepy.video.io.ImageSequenceClip
import sys
from skimage.filters import threshold_yen
from scipy.spatial.transform import Rotation
import math

# ------------------------------
# -------------------------------


def find_principal(vp1, vp2, vp3):
    # --solve AX=B using
    A = np.array([[vp1[0]-vp3[0], vp1[1]-vp3[1]],
                 [vp2[0]-vp3[0], vp2[1]-vp3[1]]])
    B = np.array([[vp2[0]*(vp1[0]-vp3[0])+vp2[1]*(vp1[1]-vp3[1])],
                  [vp1[0]*(vp2[0]-vp3[0])+vp1[1]*(vp2[1]-vp3[1])]])
    p = np.linalg.solve(A, B)
    return p[0], p[1]

# -------------------------------------------


def find_focal(px, py, vp1, vp2):

    foacal_length = np.sqrt(-px**2-py**2+(vp1[0]+vp2[0])
                            * px+(vp1[1]+vp2[1])*py-(vp1[0]*vp2[0]+vp1[1]*vp2[1]))
    return foacal_length

# -------------------------------------------


def plot_lines(src, lines, color):
    img = np.copy(src)
    for line in lines:
        x1, y1, x2, y2 = line[0:4]
        cv2.line(img, (x1, y1), (x2, y2), color, 8)
    return img

# ------------------------------------------


def get_lines(axis):
    axis_list = []

    # --set the lines manually
    if(str(axis) == "z"):
        points = np.uint(np.array([(2054.8, 2502.0), (1997.9, 794.7), (1651.7, 2497.3), (
            1594.8, 699.8), (1205.9, 225.6), (1248.6, 2132.1), (693.7, 1036.6), (679.4, 277.7)]))
    elif(str(axis) == "x"):
        points = np.uint(np.array([(1604.2, 690.3), (1988.4, 785.2), (1651.7, 2497.3), (
            2035.8, 2492.5), (3610.3, 1283.2), (4018.2, 1378.0), (3686.2, 2558.9), (4079.9, 2558.96)]))
    else:
        points = np.uint(np.array([(2832.6, 1169.3), (2073.8, 1250.0), (2775.7, 1268.9), (
            2144.9, 1325.9), (3539.2, 965.4), (3022.3, 1022.3), (3591.4, 2539.9), (3102.9, 2554.2)]))
    for i in range(len(points)//2):
        x1, y1, x2, y2 = points[2*i, 0], points[2 *
                                                i, 1], points[2*i+1, 0], points[2*i+1, 1]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        theta, slope, cons = find_slope(x1, y1, x2, y2)
        axis_list.append([x1, y1, x2, y2, theta, slope, cons])
    return axis_list
# # -------------------------
# --------------------


def find_vp(candidates):
    x_vanish = 0
    y_vanish = 0
    minimum_distance = np.inf
    for i in range(0, len(candidates)):
        for j in range(i+1, len(candidates)):
            distance_i_j = 0
            distance_i_j = np.int32(distance_i_j)
            m1, c1 = candidates[i, 5:7]
            m2, c2 = candidates[j, 5:7]
            if(m1 != m2):
                x_intersection = ((c2-c1)/(m1-m2)).astype(np.float32)
                y_intersection = (m1*(x_intersection)+c1).astype(np.float32)

                for k in range(len(candidates)):
                    m_line, c_line = candidates[k, 5:7]
                    numerator = np.abs(
                        m_line*x_intersection+c_line-y_intersection)
                    denominator = np.sqrt(1+m_line**2)
                    distance = numerator/denominator
                    distance_i_j += distance**2
                if(minimum_distance > distance_i_j):
                    x_vanish = np.copy(x_intersection)
                    y_vanish = np.copy(y_intersection)
                    minimum_distance = np.copy(distance_i_j)
    return x_vanish, y_vanish

# ----------------------------------------


def find_slope(x1, y1, x2, y2):
    # if x1 != x2, slope can be found using regular equation
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
    # if x1 = x2, slope is infinity, thus a large value
    else:
        m = 100000000
    c = y2 - m*x2  # c = y - mx from the equation of line
    # theta will contain values between -90 -> +90.
    theta = math.degrees(math.atan(m))
    return theta, m, c


# -------------------main
# -------------read image
image = cv2.imread('vns.jpg')
image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
height, width = image.shape[:2]

z_lines = get_lines("z")
new_image = plot_lines(image, z_lines, (128, 0, 128))
x_lines = get_lines("x")
new_image = plot_lines(new_image, x_lines, (255, 0, 0))
y_lines = get_lines("y")
new_image = plot_lines(new_image, y_lines, (255, 204, 0))

plt.imshow(new_image)
plt.title('close the window to continue the proccess')
plt.show()
# # # # -------------------------
# # # # -------------------------

vz_x, vz_y = find_vp(np.array(z_lines))
vx_x, vx_y = find_vp(np.array(x_lines))
vy_x, vy_y = find_vp(np.array(y_lines))

vx = np.array([vx_x, vx_y, 1])
vy = np.array([vy_x, vy_y, 1])
vz = np.array([vz_x, vz_y, 1])
px, py = find_principal(vx, vy, vz)
focal_len = find_focal(px, py, vx, vy)
# ----------------------------------
K = np.array([[float(focal_len), 0.0, float(px)],
              [0.0, float(focal_len), float(py)],
              [0.0, 0.0, 1.0]])
k_inverse = np.linalg.inv(K)

# ---backproject vanishing points
# ---to find ray direction in 3d
x_direction = np.matmul(k_inverse, vx)
y_direction = np.matmul(k_inverse, vy)

# ---cross product of two orthognal vector x and y is z
z_direction = np.cross(x_direction, y_direction)

# ----normilzing the z_direction to find angel
z_direction /= np.sqrt(z_direction[0]**2+z_direction[1]**2)

z_rotation = -1*(np.arctan2(z_direction[1], z_direction[0]) + np.pi / 2)
x_rotation = np.arcsin(z_direction[2] / np.linalg.norm(z_direction))

# ----------------------------------------------
# --------------rotation matrix  ---- res04
RX = np.array([[1,           0,            0],
               [0, np.cos(x_rotation), -np.sin(x_rotation)],
               [0, np.sin(x_rotation), np.cos(x_rotation)]])

RY = np.array([[np.cos(0), 0, np.sin(0)],
               [0, 1,            0],
               [-np.sin(0), 0, np.cos(0)]])

RZ = np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0],
               [np.sin(z_rotation), np.cos(z_rotation), 0],
               [0,            0, 1]])

# Composed rotation matrix with (RX,RY,RZ)
R = np.linalg.multi_dot([RZ, RY, RX])


H = np.linalg.multi_dot([K, R, k_inverse])
print(H)

# dc shift for better presentation
H[1, 2] += 1300
# --------------------------
res04 = cv2.warpPerspective(
    image, H, (width, height), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
# res04
plt.imshow(res04)
plt.imsave("res04.jpg", res04)
plt.show()
