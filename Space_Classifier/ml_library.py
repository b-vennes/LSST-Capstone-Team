from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import io
from skimage.transform import rescale, resize
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as img
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy
from urllib.request import urlretrieve
import os
import sys
sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

CUTOFF_VALUE = 0.5

LEARNING_RATE = 0.001

def build_binary_classifier(input_placeholder, label_placeholder, image_height, image_width, image_channels):
    # This site offers some loose guidance: https://www.tensorflow.org/tutorials/estimators/cnn

    # start our graph with the input placeholder
    graph = input_placeholder

    # add first convolution layer with 16 filters
    graph = add_convolution_layer(graph, filters=16)

    # add a pooling layer
    graph = add_pooling_layer(graph)

    # add second convolutional layer with 32 filters
    graph = add_convolution_layer(graph, filters=32)

    # add a second pooling layer
    graph = add_pooling_layer(graph)

    # add a third convolutional layer with 64 filters
    graph = add_convolution_layer(graph, filters=64)

    # flatten out so that its easy to put into one neuron
    # note: each pooling layer makes the output half as big
    # note: the 64 is the number of filters from the third convolutional layer
    graph = tf.reshape(graph, [-1, 7 * 7 * 64])

    # add two fully connected layers with dropout in betweeen
    graph = add_fully_connected_layer(graph, 1024)
    graph = add_dropout(graph)
    graph = add_fully_connected_layer(graph, 256)

    # make final guess about image
    graph = add_fully_connected_layer(graph, 1)

    # use sigmoid function to activate the neuron
    predictor = tf.nn.sigmoid(graph)

    # create optimizer using the losses
    losses = tf.losses.mean_squared_error(predictor,label_placeholder)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(losses)

    return graph, predictor, optimizer

def add_convolution_layer(input_graph, filters, kernel_size=5, padding = "SAME"):
    """
    Creates a new convolution layer using the given input, filter number, kernel size, and padding type.
    """
    
    convolution_layer = tf.layers.conv2d(
      inputs=input_graph,
      filters=filters,
      kernel_size=kernel_size,
      padding=padding,
      activation=tf.nn.relu)

    return convolution_layer

def add_pooling_layer(input_graph, pooling_size=2, num_strides=2):
    """
    Creates a new pooling layer using the given input, pooling size, and number of strides
    """

    pooling_layer = tf.layers.max_pooling2d(inputs=input_graph, pool_size=pooling_size, strides=num_strides)

    return pooling_layer

def add_fully_connected_layer(input_graph, num_units):
    """
    Creates a new fully connected layer using the given input and number of units
    """
    
    fully_connected_layer = tf.layers.dense(input_graph, units=num_units)

    return fully_connected_layer

def add_dropout(input_graph, dropout_rate=0.5):
    """

    """

    dropout = tf.layers.dropout(input_graph, rate=dropout_rate)

    return dropout


def get_accuracy(test_prediction, expected_labels):
    correct_num = 0
    
    iterator = 0
    while iterator < len(test_prediction):
        if expected_labels[iterator]:
            if test_prediction[iterator] > CUTOFF_VALUE:
                correct_num += 1
        else:
            if test_prediction[iterator] <= CUTOFF_VALUE:
                correct_num += 1
        
        iterator += 1
    
    return correct_num/len(test_prediction)


def get_confusion_matrix(test_prediction, expected_labels):
    # determine accuracy
    iterator = 0
    trousers_correct = 0
    trousers_attempts = 0
    others_correct = 0
    others_attempts = 0
    while iterator < len(test_prediction):
        if expected_labels[iterator]:
            trousers_attempts += 1

            if test_prediction[iterator] > CUTOFF_VALUE:
                trousers_correct += 1
        
        else:
            others_attempts += 1

            if test_prediction[iterator] <= CUTOFF_VALUE:
                others_correct += 1
        
        iterator += 1
    
    confusion_matrix = [trousers_correct, trousers_attempts - trousers_correct], [others_correct, others_attempts - others_correct]

    subset_accuracy = trousers_correct/trousers_attempts

    others_accuracy = others_correct/others_attempts

    return confusion_matrix, subset_accuracy, others_accuracy

def import_data():
    """
    Downloads all the image files in the database to the local images folder.
    """

    database_image_ids = DynamoConnect.get_image_ids()

    this_directory = os.path.dirname(__file__)

    local_images_file = open(os.path.join(this_directory, "Images", "image_ids.list"), "r+")
    local_image_ids = local_images_file.read().splitlines()

    for identifier in database_image_ids:

        if any(identifier in id_value for id_value in local_image_ids):
            print("identifier", identifier, "found")
            continue

        # download the image from s3
        image_link = DynamoConnect.get_image_link(identifier)

        image_name = identifier + ".fits"
        image_location = os.path.join(this_directory, "Images", image_name)

        urlretrieve(image_link, image_location)

        # save id to list file
        print(identifier,file=local_images_file)

    local_images_file.close

def load_data():
    
    this_directory = os.path.dirname(__file__)

    local_images_file = open(os.path.join(this_directory, "Images", "image_ids.list"), "r+")
    local_image_ids = local_images_file.read().splitlines()

    validation_arrays = []
    validation_labels = []
    training_arrays = []
    training_labels = []

    i = 0

    for identifier in local_image_ids:

        if i > 10:
            break
        i += 1
        image_name = identifier + ".jpg"
        image_location = os.path.join(this_directory, "Images", image_name)

        array = io.imread(image_location, as_gray=True)

        image_info = DynamoConnect.get_image_info(identifier)

        # determine the label value
        if image_info['Label'] == 'comet':
            label = 0
        else :
            label = 1

        if image_info['Validation']:
            print("Adding to validation set:", identifier)
            validation_arrays.append(array)
            validation_labels.append(label)
        else :
            print("Adding to training set:", identifier)
            training_arrays.append(array)
            training_labels.append(label)

    # return two tuples, training and validation tuples
    return (numpy.array(training_arrays, dtype=numpy.float32), training_labels), (validation_arrays, validation_labels)
