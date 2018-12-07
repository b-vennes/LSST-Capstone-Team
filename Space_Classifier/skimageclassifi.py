from sklearn import datasets, svm, metrics
from skimage import io
import matplotlib.image as img
from astropy.io import fits
import numpy

def train_model_fits():
    """
    Trains a new prediction model for classifying fits files as comets or not.
    """

    eagle_fits = fits.open('502nmos.fits')

    eagle_data = eagle_fits[0].data

    test_features = [eagle_data.flatten(), eagle_data.flatten()]

    test_targets = [0,1]

    classifier = svm.SVC(gamma=0.001)

    classifier.fit(test_features,test_targets)

def train_model(training_features, training_targets):

    clf = svm.SVC(gamma=0.001)

    # training features must be an array of data arrays
    # training targets must be an array of integer labels
    clf.fit(training_features,training_targets)

    return clf

from joblib import dump
def save_model(classifier, file_name):
    # save the model to the given file_name
    dump(classifier, file_name)

from joblib import load
def load_model(file_name):
    # load the model from the given file_name
    loaded_clf = load(file_name)
    return loaded_clf

import DynamoConnect
import urllib.request as urlreq
import os
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

        img = io.imread(image_location, as_gray=True)

        array = img.flatten()

        os.remove(image_location)

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
    return (training_arrays, training_labels), (validation_arrays, validation_labels)