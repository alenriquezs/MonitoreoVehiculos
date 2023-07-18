import cv2
from Rastreador import *

# --Vamos a crear un objeto de seguimiento----
seguimiento = Rastreador()
# --Realizamos la lectura del video-----------
cap = cv2.VideoCapture("Seguimiento de Objetos con Python_02.mp4")

# --------Vamos a realizar una deteccion de objetos con camara estable-----------
#Cambiando el tamaño del historial podemos tener mejores resultados (Camara estatica)
#Tambien modificaremos el umbral, entre menor sea, mas deteccion tendremos (Falsos Positivos)
deteccion = cv2.createBackgroundSubtractorMOG2(history = 10000, varThreshold= 12 ) #Extrae los objetos en movimiento en una camara estable


while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(1280,720))  # Redimensionamos el video

    # Elegimos una zona de interes para contar el paso de autos
    zona = frame[530:720, 270:870]

    # Creamos una mascara a los fotogramas con el fin de que nuestros objetos sean blancos y el fondo negro
    mascara = deteccion.apply(zona)
    _, mascara = cv2.threshold(mascara, 254,255, cv2.THRESH_BINARY) # Con este umbral eliminamos los pixeles grises y dejamos solo los blancos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = [] # Lista donde vamos a almacenar la info

    # Dibujamos todos los contornos en frame, de azul claro con 2 de grosor
    for cont in contornos:
        #Eliminamos los contornos pequeños
        area = cv2.contourArea(cont)
        if area > 1500: #Si el area es mayor a 100 pixeles
            x, y, ancho, alto = cv2.boundingRect(cont)
            cv2.rectangle(zona, (x,y), (x+ancho, y+alto), (255,255,0), 3) #Dibujamos el rectangulo
            detecciones.append([x, y, ancho, alto]) #Almacenamos la informacion de las detecciones

    #Seguimiento de los objetos
    info_id = seguimiento.rastreo(detecciones)
    for inf in info_id:
        x, y, ancho, alto, id = inf
        cv2.putText(zona, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        cv2.rectangle(zona, (x,y), (x+ancho, y+alto), (255,255,0), 3) #Dibujamos el rectangulo

    print(info_id)
    cv2.imshow("Zona de interes", zona)
    cv2.imshow("Carretera", frame)
    cv2.imshow("Mascara", mascara)

    key = cv2.waitKey(25)
    if key == 100:
        break

cap.release()
cv2.destroyAllWindows()