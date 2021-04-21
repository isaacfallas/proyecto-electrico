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

ventanaAncho = 10
ventanaAltura = 10
paso = 10

input_size = (224,224)

visualize = 0

min_prob = 0.75

################################################  Variables  #############################################

print("loading network...")
model = ResNet50(weights="imagenet", include_top=True)

print("reading image...")
imagen = cv2.imread('images/escarabajo2.jpg')
imageHight, imageWidth, chanels = imagen.shape

rois = []
cords = []

########################################## Obtener ROIS ####################################################

while ventanaAncho < imageWidth:

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

        cords.append((x, y, x + ventanaAncho, y + ventanaAltura))
        rois.append(roi)


#rois = np.array(rois, dtype="float32")
#predicciones = model.predict(rois)
#predicciones = imagenet_utils.decode_predictions(predicciones, top=1)
#print(predicciones)


rois = np.array(rois, dtype="float32")

##################################### Clasificacion de los ROIS ##########################################

predicciones = model.predict(rois)

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

print(labels.items())

################################## Mostrar todos los ROIS en la imagen ####################################

print("drawing all ROIS...")
for label in labels.keys():
    clone = imagen.copy()

    for(box, prob) in labels[label]:
        
        (initX, initY, endX, endY) = box
        cv2.rectangle(clone, (initX, initY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Figure", clone)
    #cv2.waitKey(0)

################################# Aplicar NMS(Non Max Suppression) ########################################
    clone = imagen.copy()
    
    boxes = np.array([p[0] for p in labels[label]])
    probability = np.array([p[1] for p in labels[label]])

    print(boxes)
    print(probability)

    boxes = non_max_suppression(boxes, probability)

    print(boxes)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Figure2", clone)
    cv2.waitKey(0)


#print(len(L))
#print(predicciones)

#labels = {}


#print(roi)

#print(cords)

#plt.figure('Escarabajo Ventana')
#plt.imshow(rois[0])
#plt.show()