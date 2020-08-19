import cv2
import os
import numpy as np
from keras.utils import np_utils

data_path = 'Dataset\dataset2'
classes = os.listdir(data_path)
labels = [i for i in range(len(classes))]

label_dict = dict(zip(classes, labels))  # empty dictionary

print(label_dict)
print(classes)
print(labels)

img_size = 100
data = []
target = []

for cls in classes:
    path = os.path.join(data_path, cls)
    img_names = os.listdir(path)

    for name in img_names:
        img_path = os.path.join(path, name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))
            data.append(resized)
            target.append(label_dict[cls])

        except Exception as e:
            print('Exception:', e)

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

new_target = np_utils.to_categorical(target)

np.save('data',data)
np.save('target',new_target)