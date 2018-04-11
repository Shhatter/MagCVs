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
positiveLister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved.xml')
chinHeightROI = 0.23
confidenceOfDetection = 0.5
imageSizeToResize = 150

haarGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
haarBadPath = "WynikiAnalizy\\Haar Cascade\\Zle\\"
lbpGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
lbpBadPath = "WynikiAnalizy\\LBP\\Zle\\"
dlibGoodPath = "WynikiAnalizy\\Dlib\\Dobre\\"
dlibBadPath = "WynikiAnalizy\\Dlib\\Zle\\"

personDefPath = "WynikiAnalizy\\ProbkiBadawcze\\Osoba"
### ZMIENNE
printDetails = True

goodResult = 0
badResult = 0

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


def removeAllResults(value):
    if value == 0:
        shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Zle\\')
        os.mkdir('WynikiAnalizy\\Haar Cascade\\Zle\\')
        shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Dobre\\')
        os.mkdir('WynikiAnalizy\\Haar Cascade\\Dobre\\')

    elif value == 1:
        shutil.rmtree('WynikiAnalizy\\LBP\\Zle\\')
        os.mkdir('WynikiAnalizy\\LBP\\Zle\\')
        shutil.rmtree('WynikiAnalizy\\LBP\\Dobre\\')
        os.mkdir('WynikiAnalizy\\LBP\\Dobre\\')
    elif value == 2:
        shutil.rmtree('WynikiAnalizy\\Dlib\\Zle\\')
        os.mkdir('WynikiAnalizy\\Dlib\\Zle\\')
        shutil.rmtree('WynikiAnalizy\\Dlib\\Dobre\\')
        os.mkdir('WynikiAnalizy\\Dlib\\Dobre\\')
    elif value == 3:
        print("Single Shot Detector ")
    elif value == 4:
        print("FaceNet")
    elif value == 10:
        print("CLEAR ALL PERSON FOLDERS !")
        for i in range(1, 11, 1):
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
            shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")

            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
            os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")


def haarCascadeFaceDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    if printDetails:
        file.writelines(
            getTime + "\t" + "Haar Cascade: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(
                scaleFactor) + "\t")
    #     file.writelines("haarFaceCascade" + "\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, scaleFactor, neighbours)
    global goodResult, badResult

    if len(detectedFace) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
        badResult += 1
    else:
        goodResult += 1
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
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def lbpCascadeDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    if printDetails:
        file.writelines(
            getTime + "\t" + "LBP: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(scaleFactor) + "\t")
    #     file.writelines("lbpCascadeDetector" + "\n\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    global goodResult, badResult
    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
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
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def dlibFaceDetector(inputFilePath, goodPath, badPath):
    if printDetails:
        file.writelines(
            getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    rects = detector(grayImage, 1)
    if len(rects) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
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
            # cv2.imshow("Output", roi_color)
            # cv2.waitKey(0)
            # show the face number
            # cv2.putText(inputFile, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(inputFile, (x, y), 1, (0, 0, 255), -1)

            # show the output image with the face detections + facial landmarks
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)


def deepLearningDetector(inputFilePath, globalConf, resizeSize):
    if printDetails:
        file.writelines("deepLearningDetector" + "\n")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    (h, w) = inputFile.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(inputFile, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
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


### Czyszczenie folderów wynikowych
# removeAllResults()


###


###

def researchModeExecutor(startOption, clear, value, lister, goodPath, badPath):
    global printDetails
    global goodResult, badResult

    if startOption == 0:
        print("HaarCascade")
        removeAllResults(0)
        counter = 0
        for image in lister:
            print(image)
            print("Iteracja: " + str(counter))
            counter += 1
            haarCascadeFaceDetector(image, value[0], value[1], goodPath, badPath)
            # lbpCascadeDetector(image, 1.5, 5)
            # dlibFaceDetector(image)
            # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
            if printDetails:
                printDetails = False
        printDetails = True
        # file.writelines(getTime + "\t")
        file.writelines("Results:\t")
        file.writelines("Good:\t" + str(goodResult) + '\t')
        file.writelines("Bad:\t" + str(badResult) + '\t')
        file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
        goodResult = 0
        badResult = 0





    elif startOption == 1:
        print("LBP")
        removeAllResults(1)
        counter = 0
        for image in lister:
            print(image)
            print("Iteracja: " + str(counter))
            counter += 1
            lbpCascadeDetector(image, value[0], value[1], goodPath, badPath)
            # lbpCascadeDetector(image, 1.5, 5)
            # dlibFaceDetector(image)
            # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
            if printDetails:
                printDetails = False
        printDetails = True
        # file.writelines(getTime + "\t")
        file.writelines("Results:\t")
        file.writelines("Good:\t" + str(goodResult) + '\t')
        file.writelines("Bad:\t" + str(badResult) + '\t')
        file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
        goodResult = 0
        badResult = 0
    elif startOption == 2:
        print("Histogram of Oriented Gradients")
        removeAllResults(2)
        counter = 0
        for image in lister:
            print(image)
            print("Iteracja: " + str(counter))
            counter += 1
            dlibFaceDetector(image, goodPath, badPath)
            # lbpCascadeDetector(image, 1.5, 5)
            # dlibFaceDetector(image)
            # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
            if printDetails:
                printDetails = False
        printDetails = True
        # file.writelines(getTime + "\t")
        file.writelines("Results:\t")
        file.writelines("Good:\t" + str(goodResult) + '\t')
        file.writelines("Bad:\t" + str(badResult) + '\t')
        file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
        goodResult = 0
        badResult = 0
    elif startOption == 3:
        print("Single Shot Detector ")
        print("Histogram of Oriented Gradients")
        removeAllResults(3)
        counter = 0
        for image in lister:
            print(image)
            print("Iteracja: " + str(counter))
            counter += 1
            # dlibFaceDetector(image, goodPath, badPath)
            # lbpCascadeDetector(image, 1.5, 5)
            # dlibFaceDetector(image)
            deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
            if printDetails:
                printDetails = False
        printDetails = True
        # file.writelines(getTime + "\t")
        file.writelines("Results:\t")
        file.writelines("Good:\t" + str(goodResult) + '\t')
        file.writelines("Bad:\t" + str(badResult) + '\t')
        file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
        goodResult = 0
        badResult = 0

    elif startOption == 4:
        print("FaceNet")


##Głowna pętla
# counter = 0
# for image in positiveLister:
#     # print(image)
#     # counter += 1
#
#     print("Iteracja: " + str(counter))
#     # print(counter)
#     # print(image)
#     # haarCascadeFaceDetector(image, 1.5, 5)
#     # lbpCascadeDetector(image, 1.5, 5)
#     dlibFaceDetector(image)
#     deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
#     if printDetails:
#         printDetails = False


# Zmiana nazewnictwa plików bez zmiany rozszerzenia
# rstrip = pathlib.Path(image).suffix
# print(rstrip)
# os.rename(image, "plik " + str(counter)+rstrip)

# TESTOWANIE HAAAR

removeAllResults(10)
getTimeFolderPersons = datetime.datetime.now()
getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d %H %M"))
# researchModeExecutor(0, 1, [2, 3], positiveLister, haarGoodPath, haarBadPath)

# for i in range(1, 11, 1):
#     scaleFactor = 2
#     neighbours = 3
#     # zwracanie listy plików w danej sciezce
#     lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
#     lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
#     lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")
#     allPathsPerPerson = [lister_good, lister_moderate, lister_bad]
#     dest_good_good = personDefPath + str(i) + "\\" + "Dobre\\" + getXTime + "HaarSF" + str(scaleFactor) + "NB" + str(
#         neighbours) + "Dobre"
#     dest_good_bad = personDefPath + str(i) + "\\" + "Dobre\\" + getXTime + "HaarSF" + str(scaleFactor) + "NB" + str(
#         neighbours) + "Zle"
#     dest_moderate_good = personDefPath + str(i) + "\\" + "Srednie\\" + getXTime + "HaarSF" + str(
#         scaleFactor) + "NB" + str(neighbours) + "Dobre"
#     dest_moderate_bad = personDefPath + str(i) + "\\" + "Srednie\\" + getXTime + "HaarSF" + str(
#         scaleFactor) + "NB" + str(neighbours) + "Zle"
#     dest_bad_good = personDefPath + str(i) + "\\" + "Zle\\" + getXTime + "HaarSF" + str(scaleFactor) + "NB" + str(
#         neighbours) + "Dobre"
#     dest_bad_bad = personDefPath + str(i) + "\\" + "Zle\\" + getXTime + "HaarSF" + str(scaleFactor) + "NB" + str(
#         neighbours) + "Zle"
#     destList = [[dest_good_good, dest_good_bad], [dest_moderate_good, dest_moderate_bad], [dest_bad_good, dest_bad_bad]]
#     iter = 0
#     for x in allPathsPerPerson:
#         os.mkdir(destList[iter][0])
#         os.mkdir(destList[iter][1])
#         researchModeExecutor(0, 1, [scaleFactor, neighbours], x, destList[iter][0], destList[iter][0])
#         iter += 1


for i in range(1, 11, 1):
    scaleFactor = 2
    neighbours = 3
    lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
    lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
    lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")
    dest_good = personDefPath + str(i) + "\\Haar\\Dobre\\"
    dest_bad = personDefPath + str(i) + "\\Haar\\Zle\\"

    file.writelines("Osoba " + str(i) + " Dobre:\t")
    researchModeExecutor(0, 1, [scaleFactor, neighbours], lister_good, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Srednie:\t")
    researchModeExecutor(0, 1, [scaleFactor, neighbours], lister_moderate, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Zle:\t")
    researchModeExecutor(0, 1, [scaleFactor, neighbours], lister_bad, dest_good, dest_bad)

for i in range(1, 11, 1):
    scaleFactor = 2
    neighbours = 3
    lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
    lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
    lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")
    dest_good = personDefPath + str(i) + "\\LBP\\Dobre\\"
    dest_bad = personDefPath + str(i) + "\\LBP\\Zle\\"

    file.writelines("Osoba " + str(i) + " Dobre:\t")
    researchModeExecutor(1, 1, [scaleFactor, neighbours], lister_good, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Srednie:\t")
    researchModeExecutor(1, 1, [scaleFactor, neighbours], lister_moderate, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Zle:\t")
    researchModeExecutor(1, 1, [scaleFactor, neighbours], lister_bad, dest_good, dest_bad)

for i in range(1, 11, 1):
    lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
    lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
    lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")
    dest_good = personDefPath + str(i) + "\\HOG\\Dobre\\"
    dest_bad = personDefPath + str(i) + "\\HOG\\Zle\\"

    file.writelines("Osoba " + str(i) + " Dobre:\t")
    researchModeExecutor(2, 1, 0, lister_good, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Srednie:\t")
    researchModeExecutor(2, 1, 0, lister_moderate, dest_good, dest_bad)
    file.writelines("Osoba " + str(i) + " Zle:\t")
    researchModeExecutor(2, 1, 0, lister_bad, dest_good, dest_bad)

# # Haar
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(0, 1, [4, 2])
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(0, 1, [1.5, 5])
#
# # LBP
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(1, 1, [4, 2])
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(1, 1, [2, 3])
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(1, 1, [1.5, 5])

# # HOG
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(2, 2, 0, positiveLister, dlibGoodPath, dlibBadPath)

# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(1, 1, [2, 3])
# getTime = str(datetime.datetime.now().ctime())
# researchModeExecutor(1, 1, [1.5, 5])

# file.writelines("LBP Stats: \t")
# file.writelines("Good:\t" + str(goodLbp) + '\t')
# file.writelines("Bad:\t" + str(badLbp) + '\n')
# file.writelines("HaarStats:\t")
# file.writelines("Good:\t" + str( goodHaar) + '\t')
# file.writelines("Bad:\t" + str(badHaar) + '\n')
# file.writelines("Dlib Stats:\t")
# file.writelines("Good:\t" + str(goodDlib) + '\t')
# file.writelines("Bad:\t" + str(badDlib) + '\n')
# file.writelines("Deep Learning Stats:\t")
# file.writelines("Good:\t" + str(goodDlib) + '\t')
# file.writelines("Bad:\t" + str(badDlib) + '\n')

file.close()
