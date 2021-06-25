# Code adapted from Musical_Genre_Classification by lelandroberts97 from https://github.com/lelandroberts97/Musical_Genre_Classification/

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
from numpy import asarray, delete
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.layers.pooling import AveragePooling2D

print("Source folder for training data:")
train_src = input()
print("Source folder for test data:")
test_src = input()

if not os.path.isdir(train_src) or not os.path.isdir(test_src):
    print("Invalid source folder(s).")
    exit(1)

# Dictionary maps the artist to an integer
label_dict = {
    'chetbaker': 0,
    'billevans': 1,
    'johncoltrane': 2,
    'mccoytyner': 3,
    'bach': 4,
    'mumfordandsons': 5,
    'gregoryalanisakov': 6,
    'mandolinorange': 7,
    'thesteeldrivers': 8,
    'bts': 9,
    'chopin': 10,
    'mamamoo': 11,
    'mozart': 12,
    'seventeen': 13,
    'tchaikovsky': 14
}

# Method to find the artist label from the name of the file
def get_text_label(file_name):
    segment_and_artist = file_name.split("_")[0]
    if segment_and_artist[1:] in label_dict:
        artist = segment_and_artist[1:]
    elif segment_and_artist[2:] in label_dict:
        artist = segment_and_artist[2:]
    elif segment_and_artist[3:] in label_dict:
        artist = segment_and_artist[3:]
    else:
        print("Invalid file name scheme.")
        exit(1)

    return artist

train_data = {'data': [], 'label': []}
for file in os.listdir(train_src):
    if file[-4:] != '.png':
        continue
    img = Image.open(train_src + "/" + file)
    label = get_text_label(file)
    arr = asarray(img)
    arr = delete(arr, 1, 2)
    train_data['data'].append(arr)
    train_data['label'].append(label)

test_data = {'data': [], 'label': []}
for file in os.listdir(test_src):
    if file[-4:] != '.png':
        continue
    img = Image.open(test_src + "/" + file)
    label = get_text_label(file)
    arr = asarray(img)
    arr = delete(arr, 1, 2)
    test_data['data'].append(arr)
    test_data['label'].append(label)

features_train = asarray(train_data['data'])
labels_train = asarray(list(map(lambda x: label_dict[x], train_data['label'])))
features_test = asarray(test_data['data'])
labels_test = asarray(list(map(lambda x: label_dict[x], test_data['label'])))

#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.15, random_state=4100)

features_train = features_train.astype('float32') / 255
features_test = features_test.astype('float32') / 255

labels_train = keras.utils.to_categorical(labels_train, 15)
labels_test = keras.utils.to_categorical(labels_test, 15)

cnn_model = keras.Sequential(name='cnn')

# Adding convolutional layer
cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(54,773,1)))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding convolutional layer
cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))

# Adding max pooling layer
cnn_model.add(AveragePooling2D(pool_size=(2,4)))

# Adding a flattened layer to input our image data
cnn_model.add(Flatten())

# Adding a dense layer with 64 neurons
cnn_model.add(Dense(64, activation='relu'))

# Adding a dropout layer for regularization
cnn_model.add(Dropout(0.25))

# Adding a dense layer with 64 neurons
cnn_model.add(Dense(64, activation='relu'))

# Adding a dropout layer for regularization
cnn_model.add(Dropout(0.25))

# Adding an output layer
cnn_model.add(Dense(15, activation='softmax'))

# Compiling our neural network
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# Fitting our neural network
history = cnn_model.fit(features_train,
                        labels_train, 
                        batch_size=32,
                        validation_data=(features_test, labels_test),
                        epochs=25)

cnn_model.save('trained_model')

score = cnn_model.evaluate(features_test, labels_test)
print("Test accuracy: ",  score[1])
