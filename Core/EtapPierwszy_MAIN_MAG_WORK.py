import glob

import dlib
import cv2
import numpy as np
# OPEN CV
import os

faceFolderPath = "Pozytywne/*"

lister = glob.glob(faceFolderPath)

# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')


###


def HaarCascadeFaceDetector(processedImage):
    dSave = 'Core/Znaleziska algorytmów wyszukiwania/Dlib'
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, 1.3, 5)
    # if detectedFace
    for (x,y,w,h) in detectedFace:
        cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

def dlibFaceDetector(processedImage):
    hSave = 'Core/Znaleziska algorytmów wyszukiwania/Haar Cascade'


def lbpCascadeDetector(processedImage):
    lsave = 'Core/Znaleziska algorytmów wyszukiwania/LBP'


for image in lister:
    HaarCascadeFaceDetector(image)
    dlibFaceDetector(image)
    lbpCascadeDetector(image)
