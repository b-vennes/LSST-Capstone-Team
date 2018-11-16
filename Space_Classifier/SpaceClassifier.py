import tensorflow as tf
from tensorflow import keras
import keras_preprocessing

import numpy as np
import matplotlib.pyplot as plt
import tkinter

import DynamoConnect
from boto3.dynamodb.conditions import Key, Attr
import boto3

import urllib.request as urlreq

import os

def classify_images():

    # import the training and test sets from the s3 and database
    (train_images, train_labels), (test_images, test_labels) = load_data()

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
    # each prediction is an array of 2 values which sum to 1
    # each index in the array contains a decimal value for the probability that the image is of the corresponding class name
    predictions = model.predict(test_images)

    print(predictions[0])

def load_data():
    """
    Downloads all the image files in the database.
    Creates a test and validation set of data and category pairs
    """

    image_ids = DynamoConnect.get_image_ids()

    training_arrays = []
    training_labels = []

    validation_arrays = []
    validation_labels = []

    for identifier in image_ids:
        # download the image from s3
        image_link = DynamoConnect.get_image_link(identifier)

        image_location = identifier + ".jpg"

        print("Image Link:", image_link)

        urlreq.urlretrieve(image_link, image_location)

        img = keras_preprocessing.image.load_img(image_location)

        # convert image to a numpy array then toss out the image file
        this_array = keras_preprocessing.image.img_to_array(img)
        os.remove(image_location)

        image_info = DynamoConnect.get_image_info(identifier)

        # determine the label value
        if image_info['Label'] == 'comet':
            this_label = 0
        else :
            this_label = 1

        if image_info['Validation']:
            print("Adding to validation set:", identifier)
            validation_arrays.append(this_array)
            validation_labels.append(this_label)
        else :
            print("Adding to training set:", identifier)
            training_arrays.append(this_array)
            training_labels.append(this_label)
        
    # return two tuples, training and validation tuples
    return (training_arrays, training_labels), (validation_arrays, validation_labels)



        

        







