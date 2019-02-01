from sklearn.linear_model import SGDClassifier
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

#General
num_channels = 1 #for grayscale

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

def build_and_train_cnn(image_arrays, image_labels, test_arrays, test_labels, image_height, image_width, image_channels):
    # This site offers some loose guidance: https://www.tensorflow.org/tutorials/estimators/cnn

    # need to restructure image data into a format that tensorflow expects (have to add an inner channel because its grayscale)
    image_arrays = numpy.expand_dims(image_arrays, axis=3)

    test_arrays = numpy.expand_dims(test_arrays, axis=3)

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_channels))
    labels_placeholder = tf.placeholder(tf.int32)

    # first convolution layer
    conv1 = tf.layers.conv2d(
      inputs=input_placeholder,
      filters=2,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

    # first pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # second convolution layer
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=4,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 4])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=2)

    predict = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits)
    }

    # calculate loss
    losses = tf.losses.sparse_softmax_cross_entropy(labels=image_labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=losses)

    # run it!
    with tf.Session() as session:
        # initialize the environment
        tf.global_variables_initializer().run()

        for i in range(0,15):
            print("run",i)
            predicted = session.run(predict, feed_dict={input_placeholder:image_arrays})
            session.run(train_op, feed_dict={input_placeholder:image_arrays})

        test_prediction = session.run(predict, feed_dict={input_placeholder:test_arrays})

    # determine accuracy
    iterator = 0
    trousers_correct = 0
    trousers_attempts = 0
    others_correct = 0
    others_attempts = 0
    while iterator < len(test_prediction["classes"]):
        if test_labels[iterator]:
            trousers_attempts += 1

            if test_prediction["classes"][iterator] == 1:
                trousers_correct += 1
        
        else:
            others_attempts += 1

            if test_prediction["classes"][iterator] == 0:
                others_correct += 1
        
        iterator += 1
    
    confusion_matrix = [trousers_correct, trousers_attempts - trousers_correct], [others_correct, others_attempts - others_correct]
    print(confusion_matrix)

def cnn_2(image_arrays, image_labels, image_height, image_width, image_channels):
    # need to restructure image data into a format that tensorflow expects (have to add an inner channel because its grayscale)
    image_arrays = numpy.expand_dims(image_arrays, axis=3)

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_channels))

    # first convolution layer - not sure about these values
    convolution_layer_1, weights1 = new_conv_layer(input_placeholder, num_channels, filter_size1, num_filters1, True)

    # second convolution layer
    convolution_layer_2, weights2 = new_conv_layer(convolution_layer_1, num_channels, filter_size2, num_filters2, True)
    
    # flatten
    layer_flat, num_features = flatten_layer(convolution_layer_2)

    # fully connected layer 1
    layer_fully1 = new_fullyconnected(layer_flat,1764,128,use_relu=True)

    # fully connected layer 2
    layer_fully2 = new_fullyconnected(layer_fully1,128,1,use_relu=False)

    # predict
    y_pred = tf.nn.softmax(layer_fully2)
    
    # return prediction
    return y_pred

# Creates new TensorFlow weights in the given shape 
# and initializing them to random values 
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	
# Creates new TensorFlow biases in the given shape 
# and initializing them to random values
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Creates a new convolutional layer in the computational graph 
def new_conv_layer(input, # previous layer
                   num_input_channels, # number of channels in input
                   filter_size, # width and height of each filter
                   num_filters, # number of filters
                   use_pooling=True): # 2x2 max-pooling

    # shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # create new weights (filters with the given shape)
    weights = new_weights(shape=shape)

    # create new biases, one for each filter
    biases = new_biases(length=num_filters)

    # create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 2, 2, 1],padding='SAME')

    # add the biases to the results of the convolution
    layer += biases

    # use pooling to down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    # rectified Linear Unit (ReLU) - calculates max(x, 0) for each input pixel x
    layer = tf.nn.relu(layer)

    # return both the resulting layer and the filter-weights
    return layer, weights

# Creates a new fully connected layer in the computational graph
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    
    # reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fullyconnected(input, num_inputs, num_outputs,use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

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

def main():
    # use the database images
    #(training_features, training_targets), (validation_features, validation_targets) = load_data()

    # use fashion set
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images/255

    trouser_not_trouser_labels = []

    test_trouser_not_trouser_labels = []

    for label in train_labels:
        if label == 1:
            trouser_not_trouser_labels.append(1)
        else:
            trouser_not_trouser_labels.append(0)

    for test_label in test_labels:
        if test_label == 1:
            test_trouser_not_trouser_labels.append(1)
        else:
            test_trouser_not_trouser_labels.append(0)

    build_and_train_cnn(train_images, trouser_not_trouser_labels, test_images, test_trouser_not_trouser_labels, 28, 28, 1)

if __name__ == "__main__":
    main()
