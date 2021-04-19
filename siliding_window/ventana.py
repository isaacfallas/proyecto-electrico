# imutils
import imutils

# numpy
import numpy as np

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

###############################  Funcion Ventana  ###############################

def ventana(imagen, paso, ventanaAnchura, ventanaAltura):
    for filas in range(0,imagen.shape[0],paso):
        for columnas in range(0,imagen.shape[1],paso):
            yield (columnas, filas, imagen[filas:ventanaAltura+filas, columnas:ventanaAnchura+columnas])

################################ Programa Prueba ################################

"""imagen = cv2.imread('../images/escarabajo1.jpg',0)

newImagen = ventana(imagen, 300, 300)
i = 0

imagenes = []

for n in newImagen:
    if i <= 1:
        imagenes.append(n)
        #plt.figure('Escarabajo Ventana')
        #plt.imshow(n)
        #plt.show()
        i=i+1

plt.figure()
plt.imshow(imagenes[0])
plt.show()

plt.figure()
plt.imshow(imagenes[1])
plt.show()
#altura, anchura = newImagen.shape
#print(altura, anchura)"""

