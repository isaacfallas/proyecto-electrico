# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# imutils
import imutils

# numpy
import numpy as np

# cv2
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

###############################  Programa Principal  ###############################

imagen = cv2.imread('../images/escarabajo1.jpg',0)

altura, anchura = imagen.shape

#print(altura,anchura)

#print(imagen[0])

#cv2.imwrite('grises.jpg',imagen)

ventana = []

for i in range(altura-100):
    ventana.append([])
    for j in range(anchura-100):
        ventana[i].append(imagen[i][j])

#for i in range(0, anchura):
#    ventana.append(imagen[i])

#print(ventana[0])

plt.figure('Escarabajo Imagen')
plt.imshow(imagen)

plt.figure('Escarabajo Ventana')
plt.imshow(ventana)
plt.show()

#imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)


#plt.figure('Escarabajo')
#plt.imshow(imagen)
#plt.show()

#imagen = imutils.opencv2matplotlib(imagen)
#print(imagen[0][0])

#imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
#print(imagen_gris[0][0])

#cv2.imshow('Gris', imagen)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#imagen = cv.findContours(imagen)

# plt.figure("Escarabajo")
# plt.imshow(imagen)
# plt.show()

#plt.figure("Correct")
#plt.imshow(imutils.opencv2matplotlib(train_images[0]))
#
#plt.show()


# plt.figure()
# plt.imshow(train_images[2])
# plt.colorbar()
# plt.grid(False)
# plt.show()