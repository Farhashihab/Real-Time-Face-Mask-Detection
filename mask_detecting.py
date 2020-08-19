from keras.models import load_model
import cv2
import numpy as np

model = load_model('Dataset\model-017.model')

face_cls = cv2.CascadeClassifier('F:\FaceMaskDetection\Dataset\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

labels_dict = {0: 'MASK', 1: ' NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

while (True):

    __, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cls.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        print(label)

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('IMAGE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
cap.release()
