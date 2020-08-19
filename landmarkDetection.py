import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("F:\FaceMaskDetection\shape_predictor_68_face_landmarks.dat")

while True:
    __, frame = cap.read()

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(img)

    for face in faces:
        print(face)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        landmarks = predictor(img,face)
        # print(landmarks)

        # for n in range(1,15):

        x1 = landmarks.part(2).x

        y1 = landmarks.part(2).y
        x2 = landmarks.part(14).x
        y2 = landmarks.part(14).y
        # print(x,y)
        # cv2.circle(frame,(x,y),3,(0,255,0),-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break