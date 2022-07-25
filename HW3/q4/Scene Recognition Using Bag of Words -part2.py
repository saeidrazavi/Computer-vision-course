import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sklearn.cluster import KMeans
import glob
import os
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics


sift = cv2.SIFT_create()
lenght_vector = np.zeros([2985])
features_space = np.zeros([0, 128])
k_means_feature = np.zeros([0, 128])
label_list = []
path_of_train_set = "Data/Train"
j = -1
# iterate all over train_set
for i, (root, dirs, files) in enumerate(os.walk(path_of_train_set)):
    if(i != 0):
        label = str(root).split("\\")[-1]
        for name in glob.glob(f'{root}/*.jpg'):
            j += 1
            # label_list.append(label)
            if(j//5 == 0):
                image = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
                kp1, des1 = sift.detectAndCompute(image, None)
                k_means_feature = np.concatenate(
                    (k_means_feature, des1), axis=0)
            else:
                continue

# ------------------------------------------
kmeans = KMeans(n_clusters=100, init='k-means++', max_iter=300,
                random_state=0).fit(k_means_feature)
# -----------------------------------------

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(kmeans.cluster_centers_)
label_list = []
label_of_train_data_set = []

hist_train = np.zeros([2985, 100]).astype(np.float16)
j = -1
start_index = 0

for i, (root, dirs, files) in enumerate(os.walk(path_of_train_set)):
    if(i != 0):

        label = str(root).split("\\")[-1]
        label_list.append(label)
        for name in glob.glob(f'{root}/*.jpg'):
            j += 1
            label_of_train_data_set.append(i)
            image = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
            kp1, des1 = sift.detectAndCompute(image, None)
            labels = neigh.kneighbors(des1, return_distance=False)
            for lebel in labels:
                hist_train[j, int(lebel)] += 1
            hist_train[j, :] = hist_train[j, :]/des1.shape[0]

# -----------------------------------
# ------------------------------------
path_of_test_set = "Data/Test"
hist_test = np.zeros([1500, 100]).astype(np.float16)
label_of_test_data_set = []
j = -1
for i, (root, dirs, files) in enumerate(os.walk(path_of_test_set)):
    if(i != 0):
        for name in glob.glob(f'{root}/*.jpg'):
            label_of_test_data_set.append(i)
            j += 1
            image = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
            kp1, des1 = sift.detectAndCompute(image, None)
            labels = neigh.kneighbors(des1, return_distance=False)
            for lebel in labels:
                hist_test[j, int(lebel)] += 1
            hist_test[j, :] = hist_test[j, :]/des1.shape[0]

k = 15
true_predication = 0
neigh = NearestNeighbors(n_neighbors=k, metric='manhattan')
neigh.fit(hist_train)
NearestNeighbors(n_neighbors=k)
indice = neigh.kneighbors(hist_test)

for i, k_nearest in enumerate(indice[1][:]):

    candidat_labels = [label_of_train_data_set[i] for i in k_nearest]
    prediceted_label = np.median(candidat_labels)

    if(label_of_test_data_set[i] == prediceted_label):
        true_predication += 1

acc = float(true_predication/1500)
print(acc*100)
