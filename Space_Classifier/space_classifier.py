import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import tkinter

def classify_images():
    # import and load the fashion dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # each image will be mapped to a single label
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # preprocess training and test sets
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # setup build layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # compile step
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # training step
    model.fit(train_images, train_labels, epochs=5)

    # evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # make predictions on the whole model
    predictions = model.predict(test_images)

    # each prediction is an array of 10 values which sum to 1
    # each index in the array contains a decimal value for the probability that the image is of the corresponding class name
