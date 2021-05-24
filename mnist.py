import numpy as np
import tensorflow as tf
from tensorflow import keras


# Network and training parameters
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
CLASSES_NUM = 10     # Number of outputs = number of digits
VALIDATION_SPLIT=0.2 # How much TRAIN is reserved for VALIDATION

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784


def process_dataset():
    # Loading MNIST dataset
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # You can verify that the split between train and test is 60,000, and 10,000 respectively.
    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)
    assert Y_train.shape == (60000,)
    assert Y_test.shape == (10000,)

    X_train = X_train.reshape(X_train.shape[0], RESHAPED)
    X_test = X_test.reshape(X_test.shape[0], RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize inputs to be within in [0, 1].
    X_train /= 255
    X_test /= 255

    return (X_train, Y_train), (X_test, Y_test)


def fit_normal():
    (X_train, Y_train), (X_test, Y_test) = process_dataset()

    # One-hot representation of the labels.
    Y_train = tf.keras.utils.to_categorical(Y_train, CLASSES_NUM)
    Y_test = tf.keras.utils.to_categorical(Y_test, CLASSES_NUM)

    # Build the model
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(CLASSES_NUM, input_shape=(RESHAPED,),
                                 name='dense_layer',
                                 activation='softmax'))

    # Summary of the model
    model.summary()

    # Compile the model
    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Make prediction
    predictions = model.predict(X_test)


if __name__ == "__main__":
    fit_normal()
