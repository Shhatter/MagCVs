import glob
import pathlib

import dlib
import cv2
import imutils
import numpy as np
import shutil
import datetime
# OPEN CV
import os
from imutils import face_utils
import argparse

from skimage import io

###STAŁE
predictor_path = "landmark/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
net = cv2.dnn.readNetFromCaffe("landmark/deploy.prototxt.txt", "landmark/res10_300x300_ssd_iter_140000.caffemodel")
faceFolderPath = "Pozytywne/*"
chinHeightROI = 0.23
confidenceOfDetection = 0.5
imageSizeToResize = 150
### ZMIENNE
printDetails = True
goodHaar = 0
goodLbp = 0
goodDlib = 0
badHaar = 0
badLbp = 0
badDlib = 0
goodDeepLearning = 0
badDeepLearning = 0
###

### Sprawdzenie czy istnieje plik do logów
getTime = str(datetime.datetime.now().ctime())
if not (pathlib.Path("LogFile.txt").is_file()):
    # os.mknod("/LogFile.txt",0)
    file = open("LogFile.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")
else:
    file = open("LogFile.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")


def removeAllResults():
    shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Zle\\')
    os.mkdir('WynikiAnalizy\\Haar Cascade\\Zle\\')
    shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Dobre\\')
    os.mkdir('WynikiAnalizy\\Haar Cascade\\Dobre\\')

    shutil.rmtree('WynikiAnalizy\\LBP\\Zle\\')
    os.mkdir('WynikiAnalizy\\LBP\\Zle\\')
    shutil.rmtree('WynikiAnalizy\\LBP\\Dobre\\')
    os.mkdir('WynikiAnalizy\\LBP\\Dobre\\')

    shutil.rmtree('WynikiAnalizy\\Dlib\\Zle\\')
    os.mkdir('WynikiAnalizy\\Dlib\\Zle\\')
    shutil.rmtree('WynikiAnalizy\\Dlib\\Dobre\\')
    os.mkdir('WynikiAnalizy\\Dlib\\Dobre\\')


def haarCascadeFaceDetector(inputFilePath, scaleFactor, neighbours):
    if printDetails:
        file.writelines("haarFaceCascade" + "\n")
        file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")
    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, scaleFactor, neighbours)
    global badHaar, goodHaar

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
        badHaar += 1
    else:
        goodHaar += 1
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Dobre\\' + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def lbpCascadeDetector(inputFilePath, scaleFactor, neighbours):
    if printDetails:
        file.writelines("lbpCascadeDetector" + "\n\n")
        file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")
    global goodLbp, badLbp
    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\LBP\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
        badLbp += 1
    else:
        goodLbp += 1
        for x, y, w, h in detectedFace:
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]
            # Pokazanie że wykrywa twarz - można pominąć

            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\LBP\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            cv2.imwrite('WynikiAnalizy\\LBP\\Dobre\\' + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def dlibFaceDetector(inputFilePath):
    if printDetails:
        file.writelines("dlibFaceDetector" + "\n")
    global badDlib, goodDlib
    inputFile = cv2.imread(inputFilePath)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    rects = detector(grayImage, 1)
    if len(rects) != 1:
        cv2.imwrite('WynikiAnalizy\\Dlib\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
        badDlib += 1
    else:
        goodDlib += 1
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(grayImage, rect)
            shape = face_utils.shape_to_np(shape)

            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            # udowodnienie że twarz wykrywa
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            if x < 0:
                x = 0
            elif x > width:
                x = width - 1

            if (y < 0):
                y = 0
            elif y > height:
                y = height - 1

            if w < 0:
                w = 0
            elif w > width:
                w = width - 1

            if (h < 0):
                h = 0
            elif h > height:
                h = height - 1

            cv2.rectangle(inputFile, (x, y), (x + w, y + h), (0, 255, 0), 2)

            smart_h = int(h * chinHeightROI)
            roi_color = inputFile[y:y + h, x:x + w]
            #
            # if smart_h > h:
            #     roi_color = inputFile[y:y + (height - 1), x:x + w]
            # else:
            #     roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + height, x:x + w]
            # croppedImage = cv2.clone
            cv2.imwrite('WynikiAnalizy\\Dlib\\Dobre\\' + pathlib.Path(inputFilePath).name, inputFile)
            # cv2.imshow("Output", roi_color)
            # cv2.waitKey(0)
            # show the face number
            # cv2.putText(inputFile, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            # for (x, y) in shape:
            #     cv2.circle(inputFile, (x, y), 1, (0, 0, 255), -1)

            # show the output image with the face detections + facial landmarks


def deepLearningDetector(inputFilePath, globalConf, resizeSize):
    if printDetails:
        file.writelines("deepLearningDetector" + "\n")
    global badDeepLearning, goodDeepLearning
    inputFile = cv2.imread(inputFilePath)
    (h, w) = inputFile.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(inputFile, (300, 300)), 1.0,
    #                              (300, 300), (104.0, 177.0, 123.0))
    inputFile = imutils.resize(inputFile, resizeSize)
    blob = cv2.dnn.blobFromImage(inputFile)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > globalConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(inputFile, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(inputFile, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow("Output", inputFile)
            cv2.waitKey(0)



### zmienne
###

# czas = datetime.datetime.now().time()

lister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved.xml')

### Czyszczenie folderów wynikowych
removeAllResults()
###


###

###Głowna pętla
counter = 0
for image in lister:
    print(image)
    counter += 1

    print("Iteracja: " + str(counter))
    # print(counter)
    # print(image)
    # haarCascadeFaceDetector(image, 1.5, 5)
    # lbpCascadeDetector(image, 1.5, 5)
    # dlibFaceDetector(image)
    deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
    if printDetails:
        printDetails = False

# Zmiana nazewnictwa plików bez zmiany rozszerzenia
# rstrip = pathlib.Path(image).suffix
# print(rstrip)
# os.rename(image, "plik " + str(counter)+rstrip)

file.writelines("LBP Stats: \n")
file.writelines("Good: " + str(goodLbp) + '\n')
file.writelines("Bad: " + str(badLbp) + '\n')
file.writelines("HaarStats: ")
file.writelines("Good: " + str(goodHaar) + '\n')
file.writelines("Bad: " + str(badHaar) + '\n')
file.writelines("Dlib Stats: ")
file.writelines("Good: " + str(goodDlib) + '\n')
file.writelines("Bad: " + str(badDlib) + '\n')
file.writelines("Deep Learning Stats: ")
file.writelines("Good: " + str(goodDlib) + '\n')
file.writelines("Bad: " + str(badDlib) + '\n')

file.close()
