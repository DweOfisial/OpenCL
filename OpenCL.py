import pyopencl as cl 
import cv2 as cv
import numpy  as np
import sys
import time

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

#Elegimos plataforma y creamos el contexto
'''Estas 3 lineas de abajo no serian necesarias ya que la funcion create_some_context lo hace sola, pero es para mostrar mas claramente la plataforma que vamos a usar
    platform= cl.get_platforms()
    gpu = cl.get_platforms()[0].get_devices()
    print(gpu)
'''

context=cl.create_some_context(interactive=True)
#creamos la cola de comando
queue = cl.CommandQueue(context)

#Creamos los buffer
mf = cl.mem_flags
img_buffer = gray.astype(np.uint8)
img_k = cl.Buffer(context,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_buffer)
#Buffer para los resultados
res = np.zeros_like(gray, dtype=np.uint8)
resultado_k = cl.Buffer(context,mf.WRITE_ONLY , res.nbytes)

#creamos el programa
start = time.time()

programa = cl.Program(context,"""
    __kernel void lumi(__global uchar *imagen, __global uchar *resultado){
        int i = get_global_id(0);
        int j = get_global_id(1);
        int size=get_global_size(1);
        //if(imagen[i*size + j] > 127){
            resultado[i * size + j] = (imagen[i * size + j] > 127) ? 255 : imagen[i * size + j];

        //}
    }
""")

try:
    programa.build()
except Exception:
    print("Error:")
    print(programa.get_build_info(context.devices[0], cl.program_build_info.LOG))
    raise


knl=programa.lumi
knl(queue, dimensiones, None, img_k, resultado_k)

#Guardamos la imagen en res
cl.enqueue_copy(queue, res, resultado_k).wait()

#Mediante numpy contamos los pixeles blancos de la imagen resultado del kernel
aux=np.sum(res==255)
end = time.time()
#Imprimimos resultados
print("Tiempo:",format(end-start),"s")
print("Pixeles blancos: ", aux)

porcentajePB=hallarPorcentaje(pixelesTotales, aux)
print("La imagen tiene una luminosidad del", porcentajePB, "%")

cv.waitKey(0)