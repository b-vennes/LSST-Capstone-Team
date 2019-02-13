import ml_library
from tensorflow import keras
from sklearn.utils import shuffle
import tensorflow as tf
import numpy
import os

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

def main():

    tf.reset_default_graph()

    input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    label_placeholder = tf.placeholder(tf.float32)

    graph, predictor, optimizer = ml_library.build_binary_classifier(input_placeholder, label_placeholder, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    # use fashion set
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    test_images = test_images/255
    test_trouser_not_trouser_labels = []

    for test_label in test_labels:
        # if test_label == 5 or test_label == 7 or test_label == 9:
        if test_label == 1:
            test_trouser_not_trouser_labels.append(1)
        else:
            test_trouser_not_trouser_labels.append(0)
    
    test_labels = test_trouser_not_trouser_labels

    test_images, test_labels = shuffle(test_images, test_labels)

    # need to restructure image data into a format that tensorflow expects (have to add an inner channel because its grayscale)
    test_images = numpy.expand_dims(test_images, axis=3)

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))
        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:test_images})
    
    accuracy = ml_library.get_accuracy(test_set_prediction, test_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, test_labels)

    print("accuracy", accuracy)
    print("confusion", confusion)

if __name__ == "__main__":
    main()