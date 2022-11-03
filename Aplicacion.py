#import tensorflow as tf

#para RaspberryPi
import tflite_runtime.interpreter as tflite

import cv2
import numpy as np
import csv
import datetime
import time



#Carga del modelo TFlite


#Path y nombre del modelo tflite
#TFLITE_MODEL = "tflite_models/modelo_caras.tflite"
#En Rasp
TFLITE_MODEL = "/usr/bin/modelo_caras.tflite"

#tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
#En Rasp
tflite_interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)

tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()





#Configuraciones

#permite utilizar la cámara
captura = cv2.VideoCapture(0)
#clasificador para buscar la posición de la cara
#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#En Rasp
faceCascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascade_frontalface_default.xml')

#variable para el tiempo





#Predicciones

#Se crea el archivo donde se lleva el registro
file = open('predicciones.csv', 'a+', newline='')
with file:
    write = csv.writer(file)
    #Header
    write.writerows([['Prediccion','HORA']]) 

#Predicción ininterrumpida de las imágenes
while True:
    #se captura cada 5s
    cv2.waitKey(5000)
    
    #lectura de la imagen
    success,img = captura.read()
    
    #se requiere que imagen esté en escala de grises
    img_grises = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #posiciones y detección de rostro(s)
    u = faceCascade.detectMultiScale(img_grises,1.1,4)
    
    #en caso de no ubicar una un rostro para que no se rompa el ciclo
    try:
        rostro = img[u[0][1]:u[0][1]+u[0][3],u[0][0]:u[0][0]+u[0][2]]
    except:
        time.sleep(5)
        
    #se requiere cambiar el formato de la imagen para la predicción
    new_ima = np.float32(rostro)
    new_ima = cv2.resize(new_ima,(224,224))
    new_ima = np.expand_dims(new_ima,axis=0)
    new_ima = new_ima/255.0
    
    #predicción
    tflite_interpreter.set_tensor(input_details[0]['index'], new_ima)
    tflite_interpreter.invoke()
    #en esta variable está la predicción
    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
    #print de la predicción
    print(tflite_model_predictions)
    #para visualizar el rostro
    #no es necesario
    cv2.imshow('webcam',rostro)
    
    max_index = np.argmax(tflite_model_predictions)
    if max_index == 0:
        max_emotion="Enojo"
    if max_index == 1:
        max_emotion="Disgusto"
    if max_index == 2:
        max_emotion="Miedo"
    if max_index == 3:
        max_emotion="Alegre"
    if max_index == 4:
        max_emotion="Triste"
    if max_index == 5:
        max_emotion="Sorpresa"
        
    fecha = datetime.datetime.now()
    hora = datetime.datetime.strftime(fecha, '%H:%M:%S')
    
    file = open('predicciones.csv', 'a+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows([[max_emotion,hora]]) 

