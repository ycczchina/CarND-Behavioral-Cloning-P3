import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images = []
measurements = []
for line in samples:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image_center = cv2.imread(current_path)

    source_path = line[1]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image_left = cv2.imread(current_path)

    source_path = line[2]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image_right = cv2.imread(current_path)

    images.extend([image_center, image_left, image_right])
    measurement_center = float(line[3])
    correction = 0.2 # this is a parameter to tune
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction
    measurements.extend([measurement_center, measurement_left, measurement_right])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

'''train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('\\')[-1]
                image_center = cv2.imread(name)
                name = './data/IMG/'+batch_sample[1].split('\\')[-1]
                image_left = cv2.imread(name)
                name = './data/IMG/'+batch_sample[2].split('\\')[-1]
                image_right = cv2.imread(name)
                measurement_center = float(batch_sample[3])

                correction = 0.2 # this is a parameter to tune
                measurement_left = measurement_center + correction
                measurement_right = measurement_center - correction
                images.extend([image_center, image_left, image_right])
                measurements.extend([measurement_center, measurement_left, measurement_right])

                # images.append(image_center)
                # measurements.append(measurement_center)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 64

train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)'''

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(rate = 0.5))
model.add(Dense(100))
model.add(Dropout(rate = 0.5))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
'''history_object = model.fit_generator(train_generator, 
	steps_per_epoch = 3 * 2 * len(train_samples), 
	validation_data = validation_generator,
    validation_steps = len(validation_samples), 
    epochs = 3, verbose = 1)'''

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')