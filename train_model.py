#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:55:31 2019

@author: norhther
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation = "relu"))
classifier.add(Dense(output_dim=1, activation="sigmoid"))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        "dataset/test",
        target_size = (64,64),
        batch_size = 32,
        class_mode = "binary"
        )

test_set = test_datagen.flow_from_directory(
        "dataset/test",
        target_size = (64,64),
        batch_size = 32,
        class_mode = "binary"
        )

from IPython.display import display

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=7,
        validation_data=test_set, validation_steps=150)


from keras.models import model_from_json
# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
