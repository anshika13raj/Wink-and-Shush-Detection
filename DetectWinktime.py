import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
import time  # Import time module for timestamps

# |----------------------------------------------------------------------------|
# detectWink
# |----------------------------------------------------------------------------|
def detectWink(frame, location, ROI, cascade):
    scaleFactor = 1.15
    neighbors = 5
    flag = 0 | cv2.CASCADE_SCALE_IMAGE
    minSize = (10, 20)
    row, col = ROI.shape
    if frame.shape != ROI.shape:
        newRow = int(row * 3 / 5)
        ROI = ROI[0:newRow, :]
    eyes = cascade.detectMultiScale(ROI, scaleFactor, neighbors, flag, minSize)
    eyes = check_box_in_box(eyes)

    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return len(eyes) == 1  # Returns True if only one eye is detected (wink detected)

# |----------------------------------------------------------------------------|
# detect
# |----------------------------------------------------------------------------|
def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.25
    minNeighbors = 4
    flag = 0
    minSize = (30, 30)

    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    detected = 0
    if len(faces) == 0:
        faceROI = gray_frame
        if detectWink(frame, (0, 0), faceROI, eyesCascade):
            detected += 1
            # Print the timestamp when a wink (blink) is detected
            print(f"Wink detected at {time.strftime('%H:%M:%S', time.localtime())}")
    else:
        faces = check_box_in_box(faces)

    for (x, y, w, h) in faces:
        faceROI = gray_frame[y:y + h, x:x + w]
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            print(f"Wink detected at {time.strftime('%H:%M:%S', time.localtime())}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

# |----------------------------------------------------------------------------|
# check_box_in_box
# |----------------------------------------------------------------------------|
def check_box_in_box(boxList):
    finalBoxList = []
    insideBoxList = []
    for index1, box1 in enumerate(boxList):
        for index2, box2 in enumerate(boxList):
            if index1 == index2:
                continue
            x1, y1, xx1, yy1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
            x2, y2, xx2, yy2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

            if not(box1.tolist() in insideBoxList):
                if x1 < x2 + 3 and y1 < y2 + 3 and xx1 > xx2 - 3 and yy1 > yy2 - 3:
                    insideBoxList.append(box2.tolist())

    for box in boxList:
        if box.tolist() not in insideBoxList:
            finalBoxList.append(box)

    return finalBoxList

# |----------------------------------------------------------------------------|
# run_on_folder
# |----------------------------------------------------------------------------| 
def run_on_folder(cascade1, cascade2, folder):
    if folder[-1] != "/":
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName is not None:
                cv2.destroyWindow(windowName)
            
            windowName = os.path.basename(f)
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            outputPath = "../winkOP/" + windowName
            print(outputPath)
            cv2.imwrite(outputPath, img)

    return totalCount

# |----------------------------------------------------------------------------|
# runonVideo
# |----------------------------------------------------------------------------|
def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while showlive:
        ret, frame = videocapture.read()
 
        if not ret:
            print("Can't capture frame")
            exit()
 
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
     
    videocapture.release()
    cv2.destroyAllWindows()

# |----------------------------------------------------------------------------|
# Main Execution
# |----------------------------------------------------------------------------|
if __name__ == "__main__":
    # Your code here

    # Check command line arguments: nothing or a folder path
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) 
              + " arguments. Expecting 0 or 1: [image-folder]")
        exit()

    # Load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                         + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                        + 'haarcascade_eye.xml')

    # Debug
    print("cv2.data.haarcascades = {}".format(cv2.data.haarcascades))

    if len(sys.argv) == 2:  # One argument provided
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of", detections, "detections")
    else:  # No arguments, run on live video
        runonVideo(face_cascade, eye_cascade)