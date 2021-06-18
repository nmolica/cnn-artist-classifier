import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from numpy import delete
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

print("Source folder for data:")
src = input()

if os.path.isfile(src):
    print("Invalid source folder.")
    exit(1)

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

data = {'data': [], 'label': []}
for file in os.listdir(src):
    if file[-4:] != '.png':
        continue
    img = Image.open(src + "/" + file)
    label = get_text_label(file)
    arr = asarray(img)
    arr = delete(arr, 3, 2)
    data['data'].append(arr)
    data['label'].append(label)

features = asarray(data['data'])
labels = asarray(list(map(lambda x: label_dict[x], data['label'])))

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.15, random_state=4100)

features_train = features_train.astype('float32') / 255
features_test = features_test.astype('float32') / 255

labels_train = keras.utils.to_categorical(labels_train, 15)
labels_test = keras.utils.to_categorical(labels_test, 15)

cnn_model = keras.Sequential(name='cnn')

# Adding convolutional layer
cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(54,773,3)))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding convolutional layer
cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding a flattened layer to input our image data
cnn_model.add(Flatten())

# Adding a dense layer with 64 neurons
cnn_model.add(Dense(64, activation='relu'))

# Adding a dropout layer for regularization
cnn_model.add(Dropout(0.25))

# Adding a second dense layer with 64 neurons
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
                        epochs=20)

train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

plt.figure(figsize = (12,8))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(test_accuracy, label='Testing Accuracy', color='red')
plt.title('Training and Testing Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
plt.xticks(range(1,16), range(1,16))
plt.savefig("results.png", bbox_inches='tight', pad_inches=0)
plt.close()

score = cnn_model.evaluate(features_test, labels_test)
print("Test accuracy: ",  score[1])