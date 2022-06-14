# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:49:20 2022

@author: Gaston Caminos
"""

#%% Identificar rostros en una imagen
import cv2
#import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('oficina.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (30, 30),   # Objetos menores a esto son ignorados
                                     maxSize = (800, 800)) # Objetos mayores a esto son ignorados

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Identificar rostros en un video

import cv2
#import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()