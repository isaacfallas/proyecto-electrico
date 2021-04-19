from ventana import ventana

# cv2
import cv2

ventanaAncho = 10
ventanaAltura = 10
paso = 10

imagen = cv2.imread('../images/escarabajo1.jpg',0)

for (x, y, ventana) in ventana(imagen, paso, ventanaAncho, ventanaAltura):
    cv2.rectangle(imagen, (x, y), (x + ventanaAncho, y + ventanaAltura), (0,255,0), 2)
    cv2.imshow('figura', imagen)
    cv2.waitKey(1)
