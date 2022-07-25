import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# ----------------
# ---functions---
# ---------------


def normilizer_matrix(src, hom):
    final = np.copy(hom)
    h, w = src.shape[:2]
    p = [[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]]
    p_prime = np.array(np.dot(hom, p))
    p_zegond = p_prime/p_prime[2, :]
    x_min = np.min(p_zegond[0, :])
    y_min = np.min(p_zegond[1, :])
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if(x_min < 0):
        t[0, 2] = x_min*-1.2
    if(y_min < 0):
        t[1, 2] = y_min*-1.2
    return np.dot(t, final)
# ------------------------------------------------


def dlt(src, dst):
    size = len(src[:, 0])
    A = np.zeros([size*2, 9])
    for i in range(0, size):
        x = src[i, 0]
        y = src[i, 1]
        x_p = dst[i, 0]
        y_p = dst[i, 1]
        temp = np.array([[-x, -y, -1, 0, 0, 0, x*y_p, y*y_p, y_p],
                         [0, 0, 0, -x, -y, -1, x*x_p, y*x_p, x_p]])
        A[2*i:2*(i+1), :] = temp
    # ----------------------------
    # svd decomposition
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = np.zeros([9, 1])
    # answer is the last column of v in svd decompostion
    v = (np.array(vh).transpose())
    row, colum = v.shape
    h = v[:, colum-1]
    homography_matrix = h.reshape(3, 3)
    homography_matrix /= homography_matrix[2, 2]
    temp = np.copy(homography_matrix[1, :])
    homography_matrix[1, :] = homography_matrix[0, :]
    homography_matrix[0, :] = temp
    return homography_matrix
# -----------------------------------------------


def ransac(src, dst):
    final_list_index = []
    itteration = 4500
    dis_threshold = 20
    accuracy = 0
    # iterate for 4500 iteration
    for iter in range(itteration):
        temp_list = []
        randomlist = random.sample(range(0, len(src)), 4)
        temp_src = np.zeros_like(src[:4, :])
        temp_dst = np.zeros_like(src[:4, :])
        # choose 4 arbitrary points
        for c, i in enumerate(randomlist):
            temp_src[c, :] = src[i, :]
            temp_dst[c, :] = dst[i, :]

        # find homography between these coresponding points
        temp_homo = dlt(temp_src, temp_dst)

        counter = 0
        # check erro rate for each candidate point
        for i in range(0, len(src[:, 1])):
            x = np.ones([3, 1])
            x[:2, 0] = src[i, :].reshape(2)
            x.reshape(3)
            x_prime = np.dot(temp_homo, x)
            x_prime = (x_prime/x_prime[2, 0])[:2, 0].reshape(1, 2)
            difference = np.sum(np.abs(x_prime[:2]-dst[i, :]))

            if(difference < dis_threshold):
                counter += 1
                temp_list.append(i)

        temp_acc = counter/len(src[:, 0])
        if(temp_acc > accuracy):
            accuracy = temp_acc
            final_list = temp_list
    return final_list

# ------------------------------------------------


# ---read images
img1 = np.array(plt.imread("im03.jpg").astype('float64'))
img2 = np.array(plt.imread("im04.jpg").astype('float64'))

# Initiate SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1.astype(np.uint8), None)
kp2, des2 = sift.detectAndCompute(img2.astype(np.uint8), None)

green_rgb = [0, 128, 0]
key1 = cv2.drawKeypoints(img1.astype(np.uint8), kp1, None, green_rgb)
key2 = cv2.drawKeypoints(img2.astype(np.uint8), kp2, None, green_rgb)

temp = np.zeros([img1.shape[0]-img2.shape[0], img2.shape[1], 3])
con_key2 = cv2.vconcat([key2.astype(np.uint8), temp.astype(np.uint8)])
res13 = cv2.hconcat([key1, con_key2])
plt.imsave("res22_corners.jpg", res13)

# -----------------------------------------
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
src_pts = np.int32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.int32([kp2[m.trainIdx].pt for m in good])

m1 = np.copy(key1)
m2 = np.copy(key2)
for point in src_pts:
    cv2.circle(m1, tuple(point), 5, (0, 0, 255), 5)

for point in dst_pts:
    cv2.circle(m2, tuple(point), 5, (0, 0, 255), 5)
temp = np.zeros([img1.shape[0]-img2.shape[0], img2.shape[1], 3])
m2 = cv2.vconcat([m2, temp.astype(np.uint8)])
res14_correspondences = cv2.hconcat([m1.astype(np.uint8), m2.astype(np.uint8)])
plt.imsave("res23_correspondences.jpg", res14_correspondences)
# --------------------------------------------------------
good = []
for m, n in matches:
    if m.distance < 0.70*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(key1.astype(np.uint8), kp1, key2.astype(
    np.uint8), kp2, good, None, [0, 0, 255], flags=2)
plt.imsave("res24_matches.jpg", img3)
# -------------------------------------------------------
# Generate 20 random numbers between 0 and len(matches)
randomlist = random.sample(range(0, len(good)), 20)
listt = []
for i in range(0, 20):
    listt.append(good[int(randomlist[i])])
img4 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(
    np.uint8), kp2, listt, None, [0, 0, 255], flags=2)
plt.imsave('res25.jpg', img4)
# --------------------------------------------------------
# -----using RANSAC
good = []
for m, n in matches:
    if m.distance < 0.70*n.distance:
        good.append(m)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
list_1 = ransac(src_pts, dst_pts)
# -----------------------------------------------
good = []
# Apply ratio test
for m, n in matches:
    if m.distance < 0.70*n.distance:
        good.append([m])

mask = np.zeros([len(good), 1])
for i in range(0, len(matches)):
    if(i in list(list_1)):
        mask[i, 0] = 1

res17 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(
    np.uint8), kp2, good, None, [0, 0, 255], flags=2)
fl = flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + \
    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
res17 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(np.uint8), kp2, good, res17, matchColor=(
    255, 0, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=fl)
plt.imsave("res26.jpg", res17)
# --------------------------------------------

src_ransac = np.zeros([len(list_1), 2])
dst_ransac = np.zeros([len(list_1), 2])

for i, number in enumerate(list_1):
    src_ransac[i, :] = src_pts[number, :]
    dst_ransac[i, :] = dst_pts[number, :]
original_homo = dlt(src_ransac, dst_ransac)
inverse_of_matrix = np.linalg.inv(original_homo)

h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, original_homo)
temp = np.copy(img2)
temp1 = cv2.vconcat([temp.astype(np.uint8), np.zeros(
    [500, img2.shape[1], 3]).astype(np.uint8)])
temp = cv2.polylines(temp1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
temp1 = cv2.vconcat([temp.astype(np.uint8), np.zeros(
    [img1.shape[0]-temp.shape[0], img2.shape[1], 3]).astype(np.uint8)])
res19 = cv2.hconcat([img1.astype(np.uint8), temp1.astype(np.uint8)])
plt.imsave("res28.jpg", res19)

# ----------------------------------------------
inverse_of_matrix = np.linalg.inv(original_homo)
inverse_of_matrix = normilizer_matrix(img2, inverse_of_matrix)
inverse_of_matrix /= inverse_of_matrix[2, 2]
dsz = (10000, 5000)
out = cv2.warpPerspective(img2, inverse_of_matrix, dsz, flags=cv2.INTER_LINEAR)
dst1 = cv2.perspectiveTransform(dst, inverse_of_matrix)
temp = np.copy(out)
temp = cv2.polylines(temp, [np.int64(dst1)], True, 255, 3, cv2.LINE_AA)

con_ver = np.zeros([temp.shape[0]-img1.shape[0], img1.shape[1], 3])
con_img1 = cv2.vconcat([img1.astype(np.uint8), con_ver.astype(np.uint8)])
res20 = cv2.hconcat([con_img1, temp.astype(np.uint8)])
plt.imsave("res29.jpg", out.astype(np.uint8))
plt.imsave("res30.jpg", res20.astype(np.uint8))
