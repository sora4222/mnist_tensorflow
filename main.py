import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sklearn.preprocessing import OneHotEncoder

tf.logging.set_verbosity(tf.logging.INFO)


LEARNING_RATE = 0.003
IMAGE_WIDTH: int = 28
IMAGE_HEIGHT: int = 28

INPUT: int = 784
LAYER_ONE: int = 200
LAYER_TWO: int = 100
LAYER_THREE: int = 60
LAYER_FOUR: int = 30
OUTPUT: int = 10

training_in_progress = None


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


def model(features, labels, mode, params):
    print("features: {}".format(features['x'].shape))
    print("labels size: {}".format(labels.shape))
    input_layer_shaped = tf.reshape(features['x'], [-1, 28, 28, 1], name="Reshape_input")

    # Initialize the layers
    layer_1_cnn_relu: tf.Tensor = tf.layers.conv2d(input_layer_shaped,
                                                   filters=32,
                                                   kernel_size=(5, 5),
                                                   strides=[1, 1],
                                                   padding='same',
                                                   activation=tf.nn.relu)

    layer_1_pooled: tf.Tensor = tf.layers.max_pooling2d(layer_1_cnn_relu,
                                                        pool_size=[2, 2],
                                                        strides=2)

    layer_2_relu: tf.Tensor = tf.layers.conv2d(layer_1_pooled,
                                               filters=64,
                                               kernel_size=(5, 5),
                                               padding='same',
                                               activation=tf.nn.relu)
    layer_2_pool: tf.Tensor = tf.layers.max_pooling2d(layer_2_relu,
                                                      pool_size=[2, 2],
                                                      strides=2)

    layer_2_pool_flat: tf.Tensor = tf.reshape(layer_2_pool,
                                              shape=[-1, layer_2_pool.shape[1] * layer_2_pool.shape[2] * 64])

    layer_3_dense: tf.Tensor = tf.layers.dense(layer_2_pool_flat,
                                               units=1024,
                                               activation=tf.nn.relu)

    layer_4_dense_dropout: tf.Tensor = tf.layers.dropout(layer_3_dense,
                                                         rate=0.4,
                                                         training=training_in_progress == tf.estimator.ModeKeys.TRAIN)

    logits_layer: tf.Tensor = tf.layers.dense(layer_4_dense_dropout,
                                              units=10)

    predictions = {
        'classes': tf.argmax(input=logits_layer, axis=1),
        'probabilities': tf.nn.softmax(logits_layer, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(labels, logits=logits_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1),
            predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == '__main__':
    # Import the datasets
    # NOTE: Image is 28 x 28 784 pixels
    train_set: np.ndarray
    train_labels: np.ndarray
    train_labels, train_set = load_and_separate_data("train.csv")
    print(train_set.shape)

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

    cnn_estimator: tf.estimator.Estimator = tf.estimator.Estimator(
        model_fn=model, model_dir="tmp/mnist_cnn")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_set},
        y=train_labels_hot,
        batch_size=400,
        num_epochs=None,
        shuffle=True)

    cnn_estimator.train(input_fn=train_input_fn,
                        steps=5,
                        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_set},
        y=train_labels_hot,
        batch_size=400,
        num_epochs=1,
        shuffle=False)

    eval_train_results = cnn_estimator.evaluate(input_fn=eval_input_fn,
                                                steps=1)
    print(eval_train_results)
