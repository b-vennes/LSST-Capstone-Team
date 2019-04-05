import os
import math
import sys

import numpy
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing
import sklearn.metrics
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime

import fits_library
import ml_library

IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def main():

    training_stars, training_nstars, validation_set, validation_labels = fits_library.parse_images(os.path.join("Images","image_ids.list"), IMAGE_HEIGHT)

    if len(training_stars) >= len(training_nstars):
        num_items = len(training_nstars)
    else:
        num_items = len(training_stars)

    # placeholder variable for our images array
    input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    label_placeholder = tf.placeholder(tf.float32)

    # get our classifier model to use for training
    graph, predictor, optimizer = ml_library.build_binary_classifier(input_placeholder, label_placeholder, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    saver = tf.train.Saver()

    # run it!
    with tf.Session() as session:

        this_directory = os.path.dirname(__file__)

        currentDT = datetime.datetime.now()

        currentDT.combine

        foldername = f'logs_{currentDT.year}{currentDT.day}{currentDT.hour}{currentDT.minute}'

        tf_output = os.path.join(this_directory, 'tf_logs', foldername)

        train_writer = tf.summary.FileWriter(tf_output, session.graph)

        merged = tf.summary.merge_all()

        # initialize the environment
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        batch_size = 3000

        num_mini_batches = int(math.ceil((2 * num_items) / batch_size))

        counter = 0

        # run through batches 100 times
        for iterator in range(500):

            print("Iterator", iterator)

            curr_star = 0
            curr_nstar = 0

            training_stars = shuffle(training_stars)
            training_nstars = shuffle(training_nstars)

            for batch_num in range(num_mini_batches):

                print("Mini Batch", batch_num)

                curr_batch = []
                curr_labels = []

                for item_count in range(int(batch_size/2)):
                    curr_batch.append(training_stars[curr_star])
                    curr_labels.append(1)
                    curr_star += 1
                    curr_batch.append(training_nstars[curr_nstar])
                    curr_labels.append(0)
                    curr_nstar += 1

                    if curr_star == num_items:
                        break

                curr_batch = numpy.stack(curr_batch)

                # normalize batch between 0 and 1
                curr_batch = (curr_batch - numpy.min(curr_batch))/numpy.ptp(curr_batch)
                curr_labels = numpy.stack(curr_labels)
                
                curr_batch, curr_labels = shuffle(curr_batch, curr_labels)

                summary, _ = session.run([merged, optimizer], feed_dict={input_placeholder:curr_batch, label_placeholder:curr_labels})

                train_writer.add_summary(summary, counter)

                counter += 1
                

        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:validation_set})

        # save the current session so that we can continue to train/predict later
        save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))

    accuracy = ml_library.get_accuracy(test_set_prediction, validation_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, validation_labels)
    f1_score_val = ml_library.get_f1_score(test_set_prediction, validation_labels)

    print("Accuracy", accuracy)
    print("Confusion", confusion)
    print("F1 Score", f1_score_val)

if __name__ == "__main__":
    main()
