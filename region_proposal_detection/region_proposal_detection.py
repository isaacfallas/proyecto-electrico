# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2

def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

labelsFilter = ["ant"]

minProb = 0.1

print("loading ResNet50 model...")

model = ResNet50(weights="imagenet")

print("loading image...")

image = cv2.imread('../images/ants.jpg')

(H, W) = image.shape[:2]

rects = selective_search(image)

proposals = []
boxes = []

print(len(rects))

for (x, y, w, h) in rects:

    if h/float(H) < 0.05 or w/float(W) < 0.05:
        continue

    if h/float(H) > 0.5 or w/float(W) > 0.5:
        continue 

    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224,224))

    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x,y,w,h))

#print(len(proposals))
#print(len(boxes))

proposals = np.array(proposals)

preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}

#print(preds)

for (i, p) in enumerate(preds):
    
    (imageID, label, prob) = p[0]
    
    if labelsFilter is not None and label not in labelsFilter:
        continue

    if prob >= minProb:
        (x, y, w, h) = boxes[i]
        box = (x, y, x+w, y+h)

        L = labels.get(label,[])
        L.append((box , prob))
        labels[label] = L

for label in labels.keys():

    clone = image.copy()

    for (box,copy) in labels[label]:
        (x1,y1,x2,y2) = box
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0,255,0), 2)

    #cv2.imshow("All posibilities", clone)
    cv2.imwrite('All posibilities.jpg',clone)

    clone = image.copy()

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    
    for (x1, y1, x2, y2) in boxes:
        
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imwrite('True posibilities.jpg',clone)
    #cv2.imshow("True posibilities", clone)
    #cv2.waitKey(0)

