import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# 1--------------------------------------------
img1 = np.array(plt.imread("im03.jpg").astype('float64'))
img2 = np.array(plt.imread("im04.jpg").astype('float64'))

# 2-------------------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1.astype(np.uint8), None)
kp2, des2 = sift.detectAndCompute(img2.astype(np.uint8), None)

green_rgb = [0, 128, 0]

test = cv2.drawKeypoints(img1.astype(np.uint8), kp1, None, green_rgb)
test2 = cv2.drawKeypoints(img2.astype(np.uint8), kp2, None, green_rgb)

temp = np.zeros([img1.shape[0]-img2.shape[0], img2.shape[1], 3])
test22 = cv2.vconcat([test2.astype(np.uint8), temp.astype(np.uint8)])
res13 = cv2.hconcat([test, test22])
plt.imsave("res13_corners.jpg", res13)

# 3-------------------------------------------
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

m1 = np.copy(test)
m2 = np.copy(test2)
for point in src_pts:
    cv2.circle(m1, tuple(point), 5, (0, 0, 255), 5)

for point in dst_pts:
    cv2.circle(m2, tuple(point), 5, (0, 0, 255), 5)
temp = np.zeros([img1.shape[0]-img2.shape[0], img2.shape[1], 3])
m2 = cv2.vconcat([m2, temp.astype(np.uint8)])
res14_correspondences = cv2.hconcat([m1.astype(np.uint8), m2.astype(np.uint8)])
plt.imsave("res14_correspondences.jpg", res14_correspondences)

# 4-----------------------------------------------
good = []
for m, n in matches:
    if m.distance < 0.65*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(test.astype(np.uint8), kp1, test2.astype(
    np.uint8), kp2, good, None, [0, 0, 255], flags=2)
plt.imsave("res15_matches.jpg", img3)

# 5---------------------------------------------
# Generate 20 random numbers between 0 and len(matches)
randomlist = random.sample(range(0, len(good)), 20)
listt = []
for i in range(0, 20):
    listt.append(good[int(randomlist[i])])
img4 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(
    np.uint8), kp2, listt, None, [0, 0, 255], flags=2)
plt.imsave('res16.jpg', img4)

# 6-------------------------------------------
MIN_MATCH_COUNT = 5
matches = bf.knnMatch(des1, des2, k=2)
good = []

# Apply ratio test
for m, n in matches:
    if m.distance < 0.65*n.distance:
        good.append(m)
if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    M, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5, maxIters=4000, confidence=0.995)
    inverse_of_matrix = np.linalg.inv(M)
    inverse_of_matrix /= inverse_of_matrix[2, 2]
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img3 = np.zeros([img1.shape[0], img2.shape[1], 3])
    img3[:img2.shape[0], :img2.shape[1], :] = np.copy(img2)
    temp = np.copy(img3)
    temp = cv2.polylines(temp, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# 7------------------------------------------
good = []
# Apply ratio test
for m, n in matches:
    if m.distance < 0.65*n.distance:
        good.append([m])
res17 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(
    np.uint8), kp2, good, None, [0, 0, 255], flags=2)
fl = flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + \
    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
res17 = cv2.drawMatchesKnn(img1.astype(np.uint8), kp1, img2.astype(np.uint8), kp2, good, res17, matchColor=(
    255, 0, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=fl)
plt.imsave("res17.jpg", res17)

# 8------------------------------------------
temp = cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
res19 = cv2.hconcat([img1.astype(np.uint8), temp.astype(np.uint8)])
plt.imsave("res19.jpg", res19)

# 9--------------------------------------------


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
        t[0, 2] = x_min*-1.2
    if(y_min < 0):
        t[1, 2] = y_min*-1.2
    return np.dot(t, final)


# 10--------------------------------------------
print(inverse_of_matrix)
new = norimilize_matrix(img2, inverse_of_matrix)
dsz = (10000, 5000)
out = cv2.warpPerspective(img2, new, dsz, flags=cv2.INTER_LINEAR)

dst1 = cv2.perspectiveTransform(dst, new)
temp = np.copy(out)
temp = cv2.polylines(temp, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA)

con_ver = np.zeros([temp.shape[0]-img1.shape[0], img1.shape[1], 3])
con_img1 = cv2.vconcat([img1.astype(np.uint8), con_ver.astype(np.uint8)])
res20 = cv2.hconcat([con_img1, temp.astype(np.uint8)])

plt.imsave("res20.jpg", out.astype(np.uint8))
plt.imsave("res21.jpg", res20.astype(np.uint8))
