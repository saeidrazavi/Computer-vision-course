'''
i train the model in google colab notebook and use trained model in this .py file for face detection
you can pass any image to this .py file  for face detection :))

 link of colab notebook  : https://colab.research.google.com/drive/1llL8OW0WTqJ421HVTngoxDolSmF796-P?usp=sharing

 link of negative dataset : https://drive.google.com/drive/folders/1ovhSaP5k2a1UNelFB8xT3IqV03QV9Wrc?usp=sharing

 link of possitive dataset : http://vis-www.cs.umass.edu/lfw/lfw.tgz

 link to trained_model for face detection : https://drive.google.com/file/d/12peFZ-zAoKt8fHFozm8zPSMKZlXuYY_6/view?usp=sharing


'''
# ----import required libraries :
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from PIL import Image
from sklearn import utils, linear_model
from skimage.feature import hog
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from sklearn import metrics
import zipfile
import os
import glob
import tarfile
import shutil
import pickle

# -------------function to detect face


def face_detection(src, model, size):
    boxes = []
    height, width = src.shape[:-1]
    n, m = size[0], size[1]

    for rate in np.arange(0.5, 2, 0.1):
        image = cv2.resize(
            src, (int(width*rate), int(height*rate)), interpolation=cv2.INTER_AREA)
        for i in range(0, int(rate*height-n), 15):
            for j in range(0, int(rate*width-m), 15):
                feature_vector = hog(image[i:i+n, j:j+m, ], orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), visualize=False, multichannel=True)
                if(model.decision_function([feature_vector]) > 850):
                    boxes.append(
                        [int(j//rate), int(i//rate), int(j//rate+128//rate), int(i//rate+128//rate)])

    return boxes

# -----------------------------------------
# ----------non maximum supression


def nms(boxes, overlapThresh):

    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# ---------------------------------------
# ---------------main code---------------
# ---------------------------------------

# load the model
loaded_model = pickle.load(open("./finalized_model3.sav", 'rb'))

original_image = plt.imread("./Persepolis.jpg")
boxes = face_detection(original_image, loaded_model, [128, 128])

nms_boxes = nms(np.array(boxes), 0.3)
img = np.copy(original_image)

for (startX, startY, endX, endY) in nms_boxes:
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Displaying the image
plt.imsave("detected_faces.jpg", img)
plt.imshow(img.astype(np.uint8))
plt.show()
