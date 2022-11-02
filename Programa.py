import numpy as np
import cv2
import tensorflow as tf 


# Carga el modelo TFLite y asigna tensores.
interpreter = tf.lite.Interpreter(model_path="tflite_models/modelo_caras.tflite")
interpreter.allocate_tensors()

# Obtener tensores de entrada y salida.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Obtiene las caracteristicas de entrada y salida (forma y tipo) para saber como debe ser la imagen de la entrada del tensor
print("\n\ninput shape:",input_details[0]['shape'])
print("input type:",input_details[0]['dtype'])
print("\noutput shape:",output_details[0]['shape'])
print("output type:",output_details[0]['dtype'],"\n\n")



# Obtiene los valores que deben tener las dimensiondes de la imagen de entrada al tensor 
_, height, width,_ = interpreter.get_input_details()[0]['shape'] 
print("Image Shape (", width, ",", height, ")\n\n")




# evita el uso de openCL y los mensajes de registro innecesarios
cv2.ocl.setUseOpenCL(False)

# diccionario que asigna a cada etiqueta una emoción (orden alfabético)
emotion_dict = {0: "Enfadado", 1: "Disgustado", 2: "Temeroso", 3: "Feliz", 4: "Triste", 5: "Sorprendido"}

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    #Busca haarcascade para dibujar un cuadro delimitador alrededor de la cara
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        #Captura la imagen cuando hay un rostro
        imagen_capturada = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (224, 224)), -1), 0)
        
        print(imagen_capturada.dtype)
        #cambia el formato de la imagen de entrada de uint8 a float32
        imagen_capturada = imagen_capturada.astype(np.float32)
        print(imagen_capturada.dtype)
        

        
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        output_index = interpreter.get_output_details()[0]['index']
        
        interpreter.set_tensor(tensor_index,imagen_capturada)  # Coloca la imagen en el tensor de entrada
        interpreter.invoke() 
        output_index = interpreter.get_output_details()[0]['index'] #Obtiene el indice de salida 
        prediccion = np.squeeze(interpreter.get_tensor(output_index)) # Obtiene el valor del tensor de salida
        
        #Coloca el texto sobre el cuadro del rostro "Se puede cambiar"
        emocion = emotion_dict[prediccion]
        cv2.putText(frame, emocion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


