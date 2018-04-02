import glob
import pathlib

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


def HaarCascadeFaceDetector(inputFile):
    dSave = 'Core/Znaleziska algorytmów wyszukiwania/Dlib'
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, 1.3, 5)
    print(detectedFace)
    print (len(detectedFace))

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Image", inputFile)
    cv2.waitKey(0)


def dlibFaceDetector(processedImage):
    hSave = 'Core/Znaleziska algorytmów wyszukiwania/Haar Cascade'


def lbpCascadeDetector(processedImage):
    lsave = 'Core/Znaleziska algorytmów wyszukiwania/LBP'


counter = 0
for image in lister:
    print(image)
    HaarCascadeFaceDetector(cv2.imread(image))
# dlibFaceDetector(image)
# lbpCascadeDetector(image)


# Zmiana nazewnictwa plików bez zmiany rozszerzenia
# rstrip = pathlib.Path(image).suffix
# print(rstrip)
# os.rename(image, "plik " + str(counter)+rstrip)
# counter += 1
