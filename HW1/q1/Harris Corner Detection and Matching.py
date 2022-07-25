# hw01-q1-computer vision
# saeed razavi 98106542

import cv2
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

import cv2
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
# -----read images
img1 = plt.imread('im01.jpg').astype('float64')
img2 = plt.imread('im02.jpg').astype('float64')

# blured image before calculating gradins
blured_image = cv2.GaussianBlur(img1, (5, 5), cv2.BORDER_DEFAULT)
blured_image2 = cv2.GaussianBlur(img2, (5, 5), cv2.BORDER_DEFAULT)


def gradian(src):
    r, g, b = cv2.split(src)

    # gradian on x axis
    r_x = cv2.Sobel(r, cv2.CV_64F, 1, 0, 3)
    g_x = cv2.Sobel(g, cv2.CV_64F, 1, 0, 3)
    b_x = cv2.Sobel(b, cv2.CV_64F, 1, 0, 3)
    grad_x = np.maximum(np.abs(r_x), np.abs(g_x), np.abs(b_x))

    # gradian on y axis
    r_y = cv2.Sobel(r, cv2.CV_64F, 0, 1, 3)
    g_y = cv2.Sobel(g, cv2.CV_64F, 0, 1, 3)
    b_y = cv2.Sobel(b, cv2.CV_64F, 0, 1, 3)
    grad_y = np.maximum(np.abs(r_y), np.abs(g_y), np.abs(b_y))

    return grad_x, grad_y


def normilizer(src):
    return (src-np.min(src))/(np.max(src)-np.min(src))


ix, iy = gradian(blured_image)
mag = np.sqrt(ix**2+iy**2)
ix_2, iy_2 = gradian(blured_image2)
mag_2 = np.sqrt(ix_2**2+iy_2**2)

plt.imsave("res01_grad.jpg", normilizer(mag), cmap='gray')
plt.imsave("res02_grad.jpg", normilizer(mag_2), cmap='gray')

sigma = 4
k = 17
sx2 = cv2.GaussianBlur(ix*ix, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)
sy2 = cv2.GaussianBlur(iy*iy, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)
sxy = cv2.GaussianBlur(ix*iy, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)
sx2_2 = cv2.GaussianBlur(ix_2*ix_2, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)
sy2_2 = cv2.GaussianBlur(iy_2*iy_2, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)
sxy_2 = cv2.GaussianBlur(ix_2*iy_2, (k, k), sigma, sigma, cv2.BORDER_DEFAULT)

det = sx2*sy2-(sxy)**2
trace = sx2+sy2
det_2 = sx2_2*sy2_2-(sxy_2)**2
trace_2 = sx2_2+sy2_2

k = 0.065
R = det-k*(trace)**2
R2 = det_2-k*(trace_2)**2
plt.imsave("res03_score.jpg", normilizer(R), cmap="gray")
plt.imsave("res04_score.jpg", normilizer(R2), cmap="gray")


norm = normilizer(R)
norm_2 = normilizer(R2)
matrix = norm > 0.41
matrix_2 = norm_2 > 0.42
plt.imsave('res05_thresh.jpg', matrix, cmap="gray")
plt.imsave('res06_thresh.jpg', matrix_2, cmap="gray")


def nms(src):
    k = 9
    result = np.copy(src)
    c1, c2 = np.array(src).shape
    for i in range(k, c1-k, 2*k):
        for j in range(k, c2-k, 2*k):
            maximum = np.max(src[i-k:i+k, j-k:j+k])
            x, y = np.where(src[i-k:i+k, j-k:j+k] == maximum)
            result[i-k:i+k, j-k:j+k] = np.zeros([2*k, 2*k])
            result[x[0] + i, y[0]+j] = maximum

    return result


nms1 = nms(matrix)
nms2 = nms(matrix_2)
res07 = np.copy(img1)
res08 = np.copy(img2)

x, y = np.where(nms1 == 1)
x1, y1 = np.where(nms2 == 1)
color = (255, 0, 0)

for i in range(0, len(x)):
    cv2.circle(res07, tuple([y[i], x[i]]), 3, color, 3)

for i in range(0, len(x1)):
    cv2.circle(res08, tuple([y1[i], x1[i]]), 3, color, 3)

plt.imsave("res07_harris.jpg", res07.astype(np.uint8))
plt.imsave("res08_harris.jpg", res08.astype(np.uint8))

# ------------------------------------------------
# exctracting feature
n = 90
feature_vector1 = []
feature_vector2 = []
x_image1, y_image1 = np.where(nms1 == 1)
x_image2, y_image2 = np.where(nms2 == 1)

for i in range(0, len(x_image1)):
    tensor_feature = img1[x_image1[i]-n:x_image1[i] +
                          n, y_image1[i]-n:y_image1[i]+n, :]
    vector = np.ravel(tensor_feature)
    feature_vector1.append(vector)

for j in range(0, len(x_image2)):
    tensor_feature2 = img2[x_image2[j]-n:x_image2[j] +
                           n, y_image2[j]-n:y_image2[j]+n, :]
    vector = np.ravel(tensor_feature2)
    feature_vector2.append(vector)
# --------------------------------------
# -------------------------------


def find_two_max(feature1, feature2):

    ratio_im1 = []
    index_im1 = []
    ratio_im2 = []
    index_im2 = []

    # first part
    for i in range(0, len(feature1)):
        min_value_first = 1e20
        min_value_second = 1e20
        min_index = 0
        for j in range(0, len(feature2)):
            dis = np.linalg.norm(feature1[i]-feature2[j])
            if(dis < min_value_first):
                min_value_second = min_value_first
                min_value_first = dis
                min_index = j
                dis -= 1
            if(min_value_first <= dis < min_value_second):
                min_value_second = dis
        ratio = float(min_value_first/min_value_second)
        ratio_im1.append(ratio)
        index_im1.append(min_index)

    # second part
    for i in range(0, len(feature2)):
        min_value_first = 1e20
        min_value_second = 1e20
        min_index = 0
        for j in range(0, len(feature1)):
            dis = np.linalg.norm(feature2[i]-feature1[j])
            if(dis < min_value_first):
                min_value_second = min_value_first
                min_value_first = dis
                min_index = j
                dis -= 1
            if(min_value_first <= dis < min_value_second):
                min_value_second = dis
        ratio = float(min_value_first/min_value_second)
        ratio_im2.append(ratio)
        index_im2.append(min_index)

    return ratio_im1, index_im1, ratio_im2, index_im2


rt1, ind1, rt2, ind2 = find_two_max(feature_vector1, feature_vector2)

# pass from Threshold filter
index1 = np.where(np.array(rt1) < (0.96))
index2 = np.where(np.array(rt2) < (0.96))
same_index = []
final_index = []

# implement the rules above to find corresponding pairs
for i in range(0, int(np.array(index1).shape[1])):
    j = ind1[int(np.array(index1)[0, i])]
    if(ind2[j] == (np.array(index1)[0, i]) and j not in same_index):
        final_index.append((np.array(index1)[0, i]))
        same_index.append(j)

# ---final result
res9 = np.copy(img1)
res10 = np.copy(img2)
color_rgb = (0, 255, 0)
for i in range(len(final_index)):
    number = int(final_index[i])
    j = ind1[number]
    res9 = cv2.circle(
        res9, (y_image1[number], x_image1[number]), radius=4, color=color_rgb, thickness=4)
    res10 = cv2.circle(
        res10, (y_image2[j], x_image2[j]), radius=4, color=color_rgb, thickness=4)
plt.imsave("res09_corres.jpg", res9.astype(np.uint8))
plt.imsave("res10_corres.jpg", res10.astype(np.uint8))

# ----------------------------

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img1.astype('int32'))
ax2.imshow(img2.astype('int32'))
for i in range(0, len(final_index)-2, 3):
    number = int(final_index[i+2])
    j = ind1[number]
    xy = (y_image1[number], x_image1[number])
    xy1 = (y_image2[j], x_image2[j])
    con = ConnectionPatch(xyA=xy, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red")
    ax2.add_artist(con)
    ax1.plot(y_image1[number], x_image1[number],
             'ro', markersize=2, color='green')
    ax2.plot(y_image2[j], x_image2[j], 'ro', markersize=2, color='green')
plt.suptitle("One third of the corresponding points connected with lines")
plt.savefig("res11.jpg")
