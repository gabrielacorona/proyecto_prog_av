import os
import tensorflow as tf
import numpy as np
import zipfile
from tensorflow import keras

local_zip = 'tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

local_zip = 'tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

training_dir = "tmp/rps/"
training_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
training_generator = training_datagen.flow_from_directory(training_dir, target_size=(150,150),class_mode='categorical')

validation_dir = "tmp/rps-test-set/"
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150,150),class_mode='categorical')

model = tf.keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
#ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

history = model.fit_generator(
    training_generator,
    epochs=25,
    validation_data = validation_generator,
    verbose = 1
)

img1 = keras.preprocessing.image.load_img(
    "tmp/rps-test-set/paper/testpaper01-00.png", target_size=(150, 150)
)
img2 = keras.preprocessing.image.load_img(
    "tmp/rps-test-set/rock/testrock01-00.png", target_size=(150, 150)
)
img3 = keras.preprocessing.image.load_img(
    "tmp/rps-test-set/scissors/testscissors01-00.png", target_size=(150, 150)
)

#TODO: Ipdate print model for multiple images and percentage
img_array = keras.preprocessing.image.img_to_array(img1)
img_array = tf.expand_dims(img_array, 0)
classes = model.predict(img_array, batch_size=10)
print(classes)
img_array = keras.preprocessing.image.img_to_array(img2)
img_array = tf.expand_dims(img_array, 0)
classes = model.predict(img_array, batch_size=10)
print(classes)
img_array = keras.preprocessing.image.img_to_array(img3)
img_array = tf.expand_dims(img_array, 0)
classes = model.predict(img_array, batch_size=10)
print(classes)

#print(model.predict([10.0]))
