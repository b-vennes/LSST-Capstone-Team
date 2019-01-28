from sklearn.linear_model import SGDClassifier
from skimage import io
from skimage.transform import rescale, resize
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy
import sys
sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

def build_and_train_cnn(image_arrays, image_labels, image_height, image_width, image_channels):
    # This site offers some loose guidance: https://www.tensorflow.org/tutorials/estimators/cnn

    # need to restructure image data into a format that tensorflow expects (have to add an inner channel array)
    image_arrays = numpy.expand_dims(image_arrays, axis=3)

    # placeholder variable  for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_channels))

    # first convolution layer with 2 filters and a kernel size of 7
    convolution_layer_1 = tf.layers.conv2d(input_placeholder, filters=3, kernel_size=7, strides=[2,2], padding="SAME")

    # build first pooling layer?
    pool_layer_1 = tf.layers.max_pooling2d(convolution_layer_1, pool_size=[2,2], strides=2)

    # build a dense layer?
    (_, pool_layer_1_height, pool_layer_1_width, pool_layer_1_channels) = pool_layer_1.shape
    flattened = tf.reshape(pool_layer_1, shape=[-1, pool_layer_1_height * pool_layer_1_width * pool_layer_1_channels])
    dense = tf.layers.dense(inputs=flattened, units=1024)
    
    # dropout?
    dropout = tf.layers.dropout(dense)
    # determine likelihood of each class
    logits = tf.layers.dense(inputs=dropout, units=2)

    # run it!
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # starts from logits layer and works its way up the chain
        output = session.run(logits, feed_dict={input_placeholder:image_arrays})

    # return some random bullshit for now
    return output

def train_sgd_model(training_features, training_targets):

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

from urllib.request import urlretrieve
import os
def import_data():
    """
    Downloads all the image files in the database to the local images folder.
    """

    database_image_ids = DynamoConnect.get_image_ids()

    this_directory = os.path.dirname(__file__)

    local_images_file = open(os.path.join(this_directory, "Images", "image_ids.list"), "r+")
    local_image_ids = local_images_file.read().splitlines()

    i = 0

    for identifier in database_image_ids:

        if i >= 10:
            break
        
        i += 1

        if any(identifier in id_value for id_value in local_image_ids):
            print("identifier", identifier, "found")
            continue

        # download the image from s3
        image_link = DynamoConnect.get_image_link(identifier)

        image_name = identifier + ".jpg"
        image_location = os.path.join(this_directory, "Images", image_name)

        urlretrieve(image_link, image_location)

        # preprocess data images
        image_data = io.imread(image_location, as_gray=True)
        image_data = resize(image_data, (28,28), anti_aliasing=True)
        io.imsave(image_location, image_data)

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

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
def main():
    # use the database images
    (training_features, training_targets), (validation_features, validation_targets) = load_data()

    garbage_pile = build_and_train_cnn(training_features, training_targets, 28, 28, 1)

    print(garbage_pile)

    #test_predictions = cross_val_predict(clf_model, validation_features, validation_targets, cv=3)

    #print("Test Predictions:", test_predictions)

    #print("Actual:", training_targets)

    #conf_matrix = confusion_matrix(validation_targets, test_predictions)

    #print(conf_matrix)

if __name__ == "__main__":
    main()
