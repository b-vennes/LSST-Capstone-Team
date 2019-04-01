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

import fits_library
import ml_library

IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def main():
    
    BATCH_SIZE = int(sys.argv[1])
    NUM_OPTIMIZATIONS_PER_BATCH = int(sys.argv[2])

    training_stars, training_nstars, validation_set, validation_labels = fits_library.parse_images(os.path.join("Images","image_ids.list"), IMAGE_HEIGHT)

    simple_validation_set = validation_set[:200]
    simple_label_set = validation_labels[:200]

    print(len(simple_validation_set))

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

    curr_star = 0
    curr_nstar = 0

    # run it!
    with tf.Session() as session:
        
        # initialize the environment
        tf.global_variables_initializer().run()

        num_batches = int(math.ceil((2 * num_items) / BATCH_SIZE))

        # num_batches = int(math.ceil(len(train_images) / BATCH_SIZE))
        print("Number of Batches:", num_batches)

        count = 1
        f1_scores_x = []
        f1_scores_y = []

        stars_x = []
        stars_y = []

        others_x = []
        others_y = []

        plt.figure(figsize=(8,6))
        plt.axis([1, num_batches * NUM_OPTIMIZATIONS_PER_BATCH, 0, 1])
        plt.xlabel("Optimization Number")
        plt.ylabel("F1 Score")
        plt.title("Validity of Predictor")

        plt.plot(f1_scores_x, f1_scores_y, "m", label="F1 Score")
        plt.plot(stars_x, stars_y, "c", label="Stars Accuracy")
        plt.plot(others_x, others_y, "r", label="Extension Accuracy")

        plt.legend()

        plt.ion()
        plt.show()

        for batch_num in range(num_batches):

            curr_batch = []
            curr_labels = []

            for item_count in range(int(BATCH_SIZE/2)):
                curr_batch.append(training_stars[curr_star])
                curr_labels.append(1)
                curr_star += 1
                curr_batch.append(training_nstars[curr_nstar])
                curr_labels.append(0)
                curr_nstar += 1

                if curr_star == num_items:
                    break

            curr_batch = numpy.stack(curr_batch)
            curr_labels = numpy.stack(curr_labels)
            
            curr_batch, curr_labels = shuffle(curr_batch, curr_labels)
            
            print("batch", batch_num + 1)

            # run the optimizer 10 times with the labels from this batch
            for opt_count in range(NUM_OPTIMIZATIONS_PER_BATCH):
                session.run(optimizer, feed_dict={input_placeholder:curr_batch, label_placeholder:curr_labels})

                curr_simple_prediction = session.run(predictor, feed_dict={input_placeholder:simple_validation_set})
                curr_f1_score = ml_library.get_f1_score(curr_simple_prediction, simple_label_set)
                curr_confusion = ml_library.get_confusion_matrix(curr_simple_prediction, simple_label_set)
                
                f1_scores_x.append(count)
                f1_scores_y.append(curr_f1_score)

                stars_x.append(count)
                stars_y.append(curr_confusion[1])

                others_x.append(count)
                others_y.append(curr_confusion[2])

                plt.plot(f1_scores_x, f1_scores_y, "m", label="F1 Score")
                plt.plot(stars_x, stars_y, "c", label="Stars Accuracy")
                plt.plot(others_x, others_y, "r", label="Extension Accuracy")

                plt.draw()
                plt.pause(0.0001)
                count += 1
                

        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:validation_set})

        # save the current session so that we can continue to train/predict later
        save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))

    accuracy = ml_library.get_accuracy(test_set_prediction, validation_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, validation_labels)
    f1_score_val = ml_library.get_f1_score(test_set_prediction, validation_labels)

    print("Accuracy", accuracy)
    print("Confusion", confusion)
    print("F1 Score", f1_score_val)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
