import ml_library
from tensorflow import keras
from sklearn.utils import shuffle
import tensorflow as tf
import numpy
import os

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

BATCH_SIZE = 100
NUM_OPTIMIZATIONS_PER_BATCH = 40

def main():

    tf.reset_default_graph()

    input_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    label_placeholder = tf.placeholder(tf.float32)

    graph, predictor, optimizer = ml_library.build_binary_classifier(input_placeholder, label_placeholder, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))

        for i in range(len(image_batches)):
            print("batch",i+1)
            # run the optimizer 10 times with the labels from this batch
            for i in range(NUM_OPTIMIZATIONS_PER_BATCH):
                print("--- optimize", i+1)
                session.run(optimizer, feed_dict={input_placeholder:image_batches[i], label_placeholder:label_batches[i]})

        # save the current session so that we can continue to train/predict later
        save_path = saver.save(session, os.path.join(os.path.dirname(__file__), 'cnn_save_state', 'cnn_model.ckpt'))

        test_set_prediction = session.run(predictor, feed_dict={input_placeholder:test_images})
    
    accuracy = ml_library.get_accuracy(test_set_prediction, test_labels)
    confusion = ml_library.get_confusion_matrix(test_set_prediction, test_labels)

    print(test_set_prediction)
    print("accuracy", accuracy)
    print("confusion", confusion)

if __name__ == "__main__":
    main()