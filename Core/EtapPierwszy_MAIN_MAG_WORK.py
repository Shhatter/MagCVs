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
from skimage import io

###STAŁE
predictor_path = "landmark/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
### ZMIENNE
printDetails = True
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
    inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
    else:
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            smart_h = int(h * 0.23)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

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
    if printDetails:
        file.writelines("dlibFaceDetector" + "\n")
        # file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n")

    inputFile = cv2.imread(processedImage)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    rects = detector(grayImage, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(grayImage, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(inputFile, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        # cv2.putText(inputFile, "Face #{}".format(i + 1), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #     cv2.circle(inputFile, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", inputFile)
        cv2.waitKey(0)

        # win = dlib.image_window()
        #
        # win.clear_overlay()
        # win.set_image(inputFile)

        # for (i, rect) in enumerate(rects):
        # shape = predictor(grayImage, rect)
        # shape = face_utils.shape_to_np(shape)
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.imshow("image",inputFile)
        # cv2.waitKey(0)

        # win.add_overlay(rect)
        # dlib.hit_enter_to_continue()

        # dets = detector(inputFile, 1)
        # for k, d in enumerate(dets):
        #     print("Number of faces detected: {}".format(len(dets)))
        #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #         k, d.left(), d.top(), d.right(), d.bottom()))
        #     # Get the landmarks/parts for the face in box d.
        #     shape = predictor(inputFile, d)
        #     print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                               shape.part(1)))


def lbpCascadeDetector(inputFilePath, scaleFactor, neighbours):
    if printDetails:
        file.writelines("lbpCascadeDetector" + "\n")
        file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")
    inputFile = cv2.imread(inputFilePath)
    inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        cv2.imwrite('WynikiAnalizy\\LBP\\Zle\\' + pathlib.Path(inputFilePath).name, inputFile)
    else:
        for x, y, w, h in detectedFace:
            smart_h = int(h * 0.23)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]
            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
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


### zmienne
faceFolderPath = "Pozytywne/*"
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
    counter += 1
    print("Iteracja: " + str(counter))
    # print(counter)
    # print(image)
    # haarCascadeFaceDetector(image, 1.5, 5)
    # lbpCascadeDetector(image, 1.5, 5)
    dlibFaceDetector(image)

    if printDetails:
        printDetails = False

# Zmiana nazewnictwa plików bez zmiany rozszerzenia
# rstrip = pathlib.Path(image).suffix
# print(rstrip)
# os.rename(image, "plik " + str(counter)+rstrip)


file.close()
