# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from siliding_window.ventana import ventana
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2

################################################  Parametros  ############################################

ventanaAncho = 300
ventanaAltura = 300
paso = 100

input_size = (224,224)

visualize = 0

min_prob = 0.7

################################################  Variables  #############################################

print("loading network...")
model = ResNet50(weights="imagenet", include_top=True)

print("reading image...")
imagen = cv2.imread('images/escarabajo1.jpg')
imageHight, imageWidth, chanels = imagen.shape

rois = []
cords = []

########################################## Obtener ROIS ####################################################

i = 0

for (x, y, roiOrig) in ventana(imagen, paso, ventanaAncho, ventanaAltura):

    roi = cv2.resize(roiOrig, input_size)
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    cords.append((x, y, x + ventanaAncho, y + ventanaAltura))
    rois.append(roi)

    if visualize==1:
        copyImage = imagen.copy()
        cv2.rectangle(copyImage, (x,y), (x+ventanaAncho, y+ventanaAltura), (0, 255, 0), 2)
        cv2.imshow("Visual", copyImage)
        cv2.imshow("ROI", roiOrig)
        cv2.waitKey(0)

    i = i + 1


rois = np.array(rois, dtype="float32")

##################################### Clasificacion de los ROIS ##########################################

predicciones = model.predict(rois)

#print(predicciones[0])

predicciones = imagenet_utils.decode_predictions(predicciones, top=1)
labels = {}


################################## ROIS con mas probabilidades ###########################################

for (i, p) in enumerate(predicciones):
    (imageID, label, prob) = p[0]

    if prob >= min_prob:
        box = cords[i]

        L = labels.get(label,[])
        L.append((box, prob))
        labels[label] = L



#print(len(L))
#print(predicciones)

#labels = {}


#print(roi)

#print(cords)

#plt.figure('Escarabajo Ventana')
#plt.imshow(rois[0])
#plt.show()