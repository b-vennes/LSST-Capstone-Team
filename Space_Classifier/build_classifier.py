import os
import math

import numpy
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras

import fits_library
import ml_library

IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
IMAGE_CHANNELS = 1

BATCH_SIZE = 50
NUM_OPTIMIZATIONS_PER_BATCH = 50

def main():

    training_stars, training_nstars, validation_set, validation_labels = fits_library.parse_images(os.path.join("Images","image_ids.list"))

    if len(training_stars) >= len(training_nstars):
        num_items = len(training_nstars)
    else:
        num_items = len(training_stars)

    print("Number of Items:", num_items)
    input()

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
        input()

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

            print(numpy.shape(curr_batch))

            curr_batch = numpy.stack(curr_batch)
            curr_labels = numpy.stack(curr_labels)
            
            curr_batch, curr_labels = shuffle(curr_batch, curr_labels)
            
            # run the optimizer 10 times with the labels from this batch
            for opt_count in range(NUM_OPTIMIZATIONS_PER_BATCH):
                print("batch", batch_num + 1, "--- optimize", opt_count+1)
                session.run(optimizer, feed_dict={input_placeholder:curr_batch, label_placeholder:curr_labels})

        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:validation_set})

        # save the current session so that we can continue to train/predict later
        # save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))

    numpy.set_printoptions(threshold=numpy.inf)
    print(test_set_prediction)
    
    with open('predictions.txt', 'w') as f:
        for p in test_set_prediction:
            f.write("%s\n" % p)

    accuracy = ml_library.get_accuracy(test_set_prediction, validation_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, validation_labels)

    print("Accuracy", accuracy)
    print("Confusion", confusion)

    if confusion[1] < 0.7 or confusion[2] < 0.7:
        print("Warning: Bad Predictor!")


if __name__ == "__main__":
    main()
