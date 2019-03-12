import os

import numpy
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras

import fits_library
import ml_library

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 1

BATCH_SIZE = 1000
NUM_OPTIMIZATIONS_PER_BATCH = 10

def main():

    train_images, train_labels, test_images, test_labels = fits_library.parse_images(os.path.join("Images","image_ids.list"))

    # shuffle the arrays consistently so that their indexes remain lined up
    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    label_placeholder = tf.placeholder(tf.float32)

    # get our classifier model to use for training
    graph, predictor, optimizer = ml_library.build_binary_classifier(input_placeholder, label_placeholder, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    saver = tf.train.Saver()

    # run it!
    with tf.Session() as session:
        # initialize the environment
        tf.global_variables_initializer().run()

        image_batches = [train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range((len(train_images) + BATCH_SIZE - 1) // BATCH_SIZE)]
        label_batches = [train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range((len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE)]

        num_batches = int(len(train_images) / BATCH_SIZE)
        print("Number of Batches:", num_batches)

        for i in range(len(image_batches)):
            print("batch",i+1)
            
            # run the optimizer 10 times with the labels from this batch
            for i in range(NUM_OPTIMIZATIONS_PER_BATCH):
                print("--- optimize", i+1)
                session.run(optimizer, feed_dict={input_placeholder:image_batches[i], label_placeholder:label_batches[i]})
            
        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:test_images})

        # save the current session so that we can continue to train/predict later
        # save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))
        
    accuracy = ml_library.get_accuracy(test_set_prediction, test_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, test_labels)

    print("Accuracy", accuracy)
    print("Confusion", confusion)


if __name__ == "__main__":
    main()
