import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sklearn.preprocessing import OneHotEncoder

LEARNING_RATE=0.00001

INPUT: int = 784
LAYER_ONE: int = 100
LAYER_TWO: int = 50
OUTPUT: int = 10

def load_and_separate_data(file_name: str, test: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    data: np.ndarray = np.loadtxt(os.path.join("MNIST_ALL", file_name), skiprows=1, delimiter=',')
    if test:
        return data

    data_labels: np.ndarray = data[:, 0]
    print("{} label shape {}".format(file_name, data_labels.shape))
    data_set: np.ndarray = data[:, 1:]
    print("{} X shape {}".format(file_name, data_set.shape))
    return np.reshape(data_labels, (-1, 1)), np.reshape(data_set, (-1, 784))


def display_random_image(dataset_X: np.ndarray, dataset_labels: np.ndarray):
    plt.figure(figsize=(8, 8))
    random_val: int = np.random.randint(0, dataset_X.shape[0])
    plt.imshow(dataset_X[random_val, :].reshape((28, 28)))
    print("The label is : {}".format(dataset_labels[random_val]))
    plt.show()

def initialize_layer():
    pass


if __name__ == '__main__':
    # Import the datasets
    # NOTE: Image is 28 x 28 784 pixels
    train_set: np.ndarray
    train_labels: np.ndarray
    train_labels, train_set = load_and_separate_data("train.csv")

    test_set: np.ndarray
    test_set = load_and_separate_data("test.csv", test=True)

    train_set = train_set / 255.0
    test_set = test_set / 255.0

    encoder: OneHotEncoder = OneHotEncoder(categories='auto', sparse=False)
    train_labels_hot = encoder.fit_transform(train_labels)
    print("One Hot Encoding:")
    print(train_labels_hot)
    print(train_labels_hot.shape)

    # Display an image
    display_random_image(train_set, train_labels)
    print("Normal labelling:")
    print(train_labels)

    input_layer: tf.Tensor = tf.placeholder(tf.float32, [None, INPUT], name="InputData")
    labels: tf.Tensor = tf.placeholder(tf.float32, [None, OUTPUT], name="LabelData")

    # Initialize the layers

    W1 = tf.get_variable("W1", shape=[INPUT, LAYER_ONE],
                         initializer=tf.contrib.layers.xavier_initializer())
    B1 = tf.get_variable("B1", shape=[1, LAYER_ONE],
                         initializer=tf.ones_initializer())
    # Multiply and activate the layers

    # Initialize the next

    W2 = tf.get_variable("W2", shape=[LAYER_ONE,  LAYER_TWO],
                         initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.get_variable("B2", shape=[1, LAYER_TWO],
                         initializer=tf.ones_initializer())

    # Create another layer with logits
    Z1 = tf.matmul(input_layer, W1) + B1
    A1 = tf.nn.relu(Z1, name="Relu-Activation1")

    Z2 = tf.matmul(A1, W2) + B2
    A2 = tf.nn.relu(Z2, name="Relu-Activation2")

    W3 = tf.get_variable("W3", shape=[LAYER_TWO, OUTPUT],
                         initializer=tf.contrib.layers.xavier_initializer())
    B3 = tf.get_variable("B3", shape=[1, OUTPUT],
                         initializer=tf.ones_initializer())
    output_logits = tf.matmul(A2, W3) + B3

    y_vals = tf.nn.softmax(output_logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits,
                                                               labels=train_labels,
                                                               name="Cross-Entropy")

    cost_op = tf.reduce_mean(cross_entropy, name="Loss-function")

    # Optimize the logits
    correct_prediction = tf.equal(tf.argmax(y_vals, 1), tf.argmax(train_labels, 1))
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_op)

    accuracy: tf.Tensor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create summaries of the variables
    tf.summary.scalar("Loss_function", cost_op)
    tf.summary.scalar("Accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    # Predict the outputs
    with tf.Session(graph=tf.get_default_graph())as sess:
        writer = tf.summary.FileWriter("tensorboardlog",
                                       graph=tf.get_default_graph())
        writer.add_graph(tf.get_default_graph())
        sess.run(init)
        for i in range(0, 10):
            train_result, summary, accuracy_val = sess.run([train_op, summary_op, accuracy],
                                                           feed_dict={input_layer: train_set,
                                                                      labels: train_labels_hot})

            writer.add_summary(summary, i)
            print("Epoch: {}, {:.3f}%".format(i, accuracy_val * 100))

