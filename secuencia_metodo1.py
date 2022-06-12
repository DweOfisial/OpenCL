import cv2 as cv
import numpy  as np
import sys
import time 

#Funcion que halla el porcentaje de pixeles blancos en la imagen
def hallarPorcentaje(total, pixelBlancos):
    porcentaje = (pixelBlancos*100)/total
    return porcentaje

#Seleccionamos la imagen que queramos 
imagen = cv.imread('imagenes/LosAngeles.png')

gray=cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', gray)                    #Si queremos que muestre la imagen en blanco y negro descomentamos esta linea

dimensiones=gray.shape
print ("Resolucion:",dimensiones[0],"x ",dimensiones[1])

pixelesTotales=(dimensiones[0]*dimensiones[1])
print("Pixeles totales:", pixelesTotales)

#Contamos el numero de pixeles blancos
pB=0
#Valor a partir del cual decimos que el pixel esta iluminado
luminoso=127

start = time.time()

for i in range(dimensiones[0]):
    for j in range(dimensiones[1]):
        if gray[i,j] > luminoso:           
            pB=pB+1
        
end = time.time()

print("Tiempo:",format(end-start),"s")
print("Pixeles blancos: ", pB)

porcentajePB=hallarPorcentaje(pixelesTotales, pB)
print("La imagen tiene una luminosidad del", porcentajePB, "%")

cv.waitKey(0)