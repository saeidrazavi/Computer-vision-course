import zipfile
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import NearestNeighbors


# --------------------------------
size_of_resized_images = (8, 8)
labels_list = []  # --label list for each image of train set
features_vector = []  # -- flatten feature vector for each image
path_of_train_set = "Data/Train"

# iterate all over train_set
for i, (root, dirs, files) in enumerate(os.walk(path_of_train_set)):
    if(i != 0):
        label = str(root).split("\\")[-1]
        for name in glob.glob(f'{root}/*.jpg'):
            labels_list.append(i)
            image = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
            resized = np.array(cv2.resize(
                image, size_of_resized_images, interpolation=cv2.INTER_AREA))
            features_vector.append(np.ravel(resized).astype(np.int32))

# iterate all over test set

k = 1
true_predication = 0
neigh = NearestNeighbors(n_neighbors=k, metric='manhattan')
neigh.fit(features_vector)
NearestNeighbors(n_neighbors=k)

number_of_true_prediction = 0
path_of_test_set = "Data/Test"
for i, (root, dirs, files) in enumerate(os.walk(path_of_test_set)):
    if(i != 0):
        label = str(root).split("/")[-1]
        for name in glob.glob(f'{root}/*.jpg'):
            image = image = cv2.imread(str(name), cv2.IMREAD_GRAYSCALE)
            resized = np.array(cv2.resize(
                image, size_of_resized_images, interpolation=cv2.INTER_AREA))
            feature = np.ravel(resized).astype(np.int32)
            # -----------------------------
            # -------------measure accuracy
            indice = neigh.kneighbors(feature.reshape(1, -1))[1]
            candidat_labels = [labels_list[ii] for ii in indice[0]]
            prediceted_label = np.median(candidat_labels)
            if(i == prediceted_label):
                true_predication += 1
acc = float(true_predication/1500)

print(acc*100)
