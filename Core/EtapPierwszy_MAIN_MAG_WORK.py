import glob
import pathlib

import dlib
import cv2
import numpy as np
import shutil
import datetime
# OPEN CV
import os


def removeAllResults():
    shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Zle\\')
    os.mkdir('WynikiAnalizy\\Haar Cascade\\Zle\\')
    shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Dobre\\')
    os.mkdir('WynikiAnalizy\\Haar Cascade\\Dobre\\')

    shutil.rmtree('WynikiAnalizy\\LBP\\Zle\\')
    os.mkdir('WynikiAnalizy\\LBP\\Zle\\')
    shutil.rmtree('WynikiAnalizy\\LBP\\Dobre\\')
    os.mkdir('WynikiAnalizy\\LBP\\Dobre\\')


def HaarCascadeFaceDetector(inputFilePath):
    inputFile = cv2.imread(inputFilePath)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, 1.3, 5)

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
    else:
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            roi_color = inputFile[y:y + h + int(h * 0.23), x:x + w]
            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def dlibFaceDetector(processedImage):
    hSave = 'Core/Znaleziska algorytmów wyszukiwania/Haar Cascade'


def lbpCascadeDetector(inputFilePath):
    inputFile = cv2.imread(inputFilePath)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, 1.5, 5)

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\LBP\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
    else:
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            roi_color = inputFile[y:y + h + int(h * 0.23), x:x + w]
            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            cv2.imwrite('WynikiAnalizy\\LBP\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


faceFolderPath = "Pozytywne/*"
# czas = datetime.datetime.now().time()

lister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved.xml')
# Czyszczenie folderów wynikowych

removeAllResults()
counter = 0
for image in lister:
    # print(counter)
    # print(image)
    HaarCascadeFaceDetector(image)
    # dlibFaceDetector(image)
    lbpCascadeDetector(image)

# Zmiana nazewnictwa plików bez zmiany rozszerzenia
# rstrip = pathlib.Path(image).suffix
# print(rstrip)
# os.rename(image, "plik " + str(counter)+rstrip)
# counter += 1
