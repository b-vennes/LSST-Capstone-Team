from sklearn.linear_model import SGDClassifier
from skimage import io
import matplotlib.image as img
from astropy.io import fits
import numpy
import PIL.Image as Image

def train_model(training_features, training_targets):

    clf = SGDClassifier()

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
from urllib.request import urlretrieve
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

    retrieve_count = 0

    for identifier in image_ids:
        # download the image from s3
        image_link = DynamoConnect.get_image_link(identifier)

        image_location = identifier + ".jpg"

        print("Image Link:", image_link)

        urlretrieve(image_link, image_location)

        # resize image to standard size
        Image.open(image_location).resize((20,20), Image.ANTIALIAS).save(image_location)

        array = io.imread(image_location, as_gray=True).flatten()

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

        # for now retrieve 4 images
        if retrieve_count == 20:
            break
        else:
            retrieve_count += 1
        
    # return two tuples, training and validation tuples
    return (training_arrays, training_labels), (validation_arrays, validation_labels)


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
def main():
    # use the database images
    (training_features,training_targets), (validation_features, validation_targets) = load_data()

    clf_model = train_model(training_features,training_targets)

    test_predictions = cross_val_predict(clf_model, training_features, training_targets, cv=3)

    print("Test Predictions:", test_predictions)

    print("Actual:", training_targets)

    conf_matrix = confusion_matrix(training_targets, test_predictions)

    print(conf_matrix)

if __name__ == "__main__":
    main()