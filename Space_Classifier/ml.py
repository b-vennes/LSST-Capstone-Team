from sklearn.linear_model import SGDClassifier
from skimage import io
from skimage.transform import rescale, resize
import tensorflow as tf
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

def build_and_train_cnn(image_arrays, image_labels, image_height, image_width, image_channels):
    # This site offers some loose guidance: https://www.tensorflow.org/tutorials/estimators/cnn

    # need to restructure image data into a format that tensorflow expects (have to add an inner channel because its grayscale)
    image_arrays = numpy.expand_dims(image_arrays, axis=3)

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_channels))

    # first convolution layer with 3 filters and a kernel size of 7
    # strides?
    convolution_layer_1 = tf.layers.conv2d(input_placeholder, filters=3, kernel_size=7, strides=[2,2], padding="SAME")

    # build first pooling layer?
    pool_layer_1 = tf.layers.max_pooling2d(convolution_layer_1, pool_size=[2,2], strides=2)

    # build a dense layer?
    (_, pool_layer_1_height, pool_layer_1_width, pool_layer_1_channels) = pool_layer_1.shape
    flattened = tf.reshape(pool_layer_1, shape=[-1, pool_layer_1_height * pool_layer_1_width * pool_layer_1_channels])
    dense = tf.layers.dense(inputs=flattened, units=1024)
    
    # dropout?
    dropout = tf.layers.dropout(dense, rate=0.4)
    # determine likelihood of each class (either a star or not)
    logits = tf.layers.dense(inputs=dropout, units=2)

    # make predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.sparse_softmax_cross_entropy_with_logits(labels=image_labels, logits=logits)
    }

    # determine loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=image_labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

    # run it!
    with tf.Session() as session:
        # initialize the environment
        tf.global_variables_initializer().run()

        # run the optimizer 10000 times!
        for i in range(10000):
            print(i)
            session.run(optimizer, feed_dict={input_placeholder:image_arrays})
        
        # get predictions!
        output = session.run(predictions, feed_dict={input_placeholder:image_arrays})

    # for now just return the results I guess
    return output

def cnn(image_arrays, image_labels, image_height, image_width, image_channels):
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
    (training_features, training_targets), (validation_features, validation_targets) = load_data()

    garbage_pile = build_and_train_cnn(training_features, training_targets, 28, 28, 1)

    print(garbage_pile)

if __name__ == "__main__":
    main()
