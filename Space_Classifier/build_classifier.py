import ml_library
from tensorflow import keras
from sklearn.utils import shuffle
import tensorflow as tf
import numpy
import os

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

BATCH_SIZE = 1000
NUM_OPTIMIZATIONS_PER_BATCH = 10

def main():

    # use fashion set
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images/255

    test_images = test_images/255

    trouser_not_trouser_labels = []

    test_trouser_not_trouser_labels = []

    for label in train_labels:
        # if label == 5 or label == 7 or label == 9:
        if label == 1:
            trouser_not_trouser_labels.append(1)
        else:
            trouser_not_trouser_labels.append(0)

    for test_label in test_labels:
        # if test_label == 5 or test_label == 7 or test_label == 9:
        if test_label == 1:
            test_trouser_not_trouser_labels.append(1)
        else:
            test_trouser_not_trouser_labels.append(0)

    train_labels = trouser_not_trouser_labels
    test_labels = test_trouser_not_trouser_labels

    # shuffle the arrays consistently so that their indexes remain lined up
    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    # need to restructure image data into a format that tensorflow expects (have to add an inner channel because its grayscale)
    train_images = numpy.expand_dims(train_images, axis=3)

    train_labels = numpy.reshape(train_labels, newshape=(len(train_labels), 1))

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    label_placeholder = tf.placeholder(tf.float32)

    graph, predictor, optimizer = ml_library.build_binary_classifier(input_placeholder, label_placeholder, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    saver = tf.train.Saver()

    # run it!
    with tf.Session() as session:
        # initialize the environment
        tf.global_variables_initializer().run()

        image_batches = [train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range((len(train_images) + BATCH_SIZE - 1) // BATCH_SIZE)]
        label_batches = [train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range((len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE)]

        for i in range(len(image_batches)):
            print("batch",i+1)
            # run the optimizer 10 times with the labels from this batch
            for i in range(NUM_OPTIMIZATIONS_PER_BATCH):
                print("--- optimize", i+1)
                session.run(optimizer, feed_dict={input_placeholder:image_batches[i], label_placeholder:label_batches[i]})

        # save the current session so that we can continue to train/predict later
        save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))


if __name__ == "__main__":
    main()