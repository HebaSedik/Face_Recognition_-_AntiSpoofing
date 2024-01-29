import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
from ultralytics import YOLO
import cvzone
import math
import pygame
#import serial

# Open the serial port (update the port name based on your Arduino connection)
#arduino = serial.Serial('COM5', 9600)

path = 'home'
images = []
classNames1 = []
myList = os.listdir(path)
print(myList)
confidance = 0.7
cap = cv2.VideoCapture(0)
cap.set(3, 2000)
cap.set(4, 1080)
model = YOLO('n_version_1_5.pt')
classNames = ["fake", "real"]
prev_frame_time = 0
new_frame_time = 0
confidence_threshold = 0.7
previous_detection = None
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('1.mp3')

encodeListKnown = []
classNamesKnown = []

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames1.append(os.path.splitext(cl)[0])
    img = face_recognition.load_image_file(f'{path}/{cl}')
    encode = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(encode)
    classNamesKnown.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('home.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
#print('Encoding Complete')
cap = cv2.VideoCapture(0)
frame_resizing = 0.25

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    fake_detected = False

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames1[matchIndex].upper()
            print(name)
            faceLoc = np.array(faceLoc)
            faceLoc = faceLoc / 0.25
            faceLoc = faceLoc.astype(int)
            y1, x2, y2, x1 = faceLoc[0], faceLoc[1], faceLoc[2], faceLoc[3]
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > confidence_threshold:
                        if classNames[cls] == 'real':
                            color = (0, 255, 0)
                            if previous_detection == 'fake':
                                pygame.mixer.Sound.stop(alarm_sound)
                                #arduino.write(b'0')
                            # Send '1' to turn on the LED when a real and known person is detected
                            #arduino.write(b'1')
                        else:
                            color = (0, 0, 255)
                            fake_detected = True
                            pygame.mixer.Sound.play(alarm_sound)
                            #arduino.write(b'0')
                        #if classNames[cls] == 'no detection':
                            #arduino.write(b'0')
                        cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (128, 0, 128), cv2.FILLED)
                        cv2.putText(img, name + " " + f'{classNames[cls].upper()} {int(conf * 100)}%', (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNamesKnown[matchIndex].upper()
            faceLoc = np.array(faceLoc)
            faceLoc = faceLoc * 4
            faceLoc = faceLoc.astype(int)
            y1, x2, y2, x1 = faceLoc[0], faceLoc[1], faceLoc[2], faceLoc[3]

        else:
            name = "Unknown"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            pygame.mixer.Sound.play(alarm_sound)
            # Send '0' to turn off the LED when an unknown or fake face is detected
            #arduino.write(b'0')

    if fake_detected:
        previous_detection = 'fake'
    else:
        previous_detection = 'real'

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break

# Close the serial port when done
arduino.close()
cap.release()
cv2.destroyAllWindows()
