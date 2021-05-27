import numpy as np
import tensorflow as tf
from tensorflow import keras
from generator import DataSequence, data_generator


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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # You can verify that the split between train and test is 60,000, and 10,000 respectively.
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    x_train = x_train.reshape(x_train.shape[0], RESHAPED)
    x_test = x_test.reshape(x_test.shape[0], RESHAPED)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize inputs to be within in [0, 1].
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def fit_normal():
    (x_train, y_train), (x_test, y_test) = process_dataset()

    # One-hot representation of the labels.
    y_train = tf.keras.utils.to_categorical(y_train, CLASSES_NUM)
    y_test = tf.keras.utils.to_categorical(y_test, CLASSES_NUM)

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
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Make prediction
    predictions = model.predict(x_test)


def fit_sequence():
    (x_train, y_train), (x_test, y_test) = process_dataset()

    pivot = int((1-VALIDATION_SPLIT)*len(y_train))
    x_train, x_validation = x_train[:pivot], x_train[pivot:]
    y_train, y_validation = y_train[:pivot], y_train[pivot:]

    training_sequence = DataSequence(x_train, y_train, CLASSES_NUM, BATCH_SIZE)
    validation_sequence = DataSequence(x_validation, y_validation, CLASSES_NUM, BATCH_SIZE)

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
    # y
    #   If `x` is a dataset, generator, or keras.utils.Sequence instance,
    #   `y` should not be specified (since targets will be obtained from x).
    # batch_size
    #   Do not specify the batch_size if your data is in the form of datasets,
    #   generators, or keras.utils.Sequence instances (since they generate batches).
    # validation_split
    #   This parameter is not supported when x is a dataset, generator or
    #   keras.utils.Sequence instance.
    model.fit(training_sequence,
              validation_data=validation_sequence,
              epochs=EPOCHS,
              verbose=VERBOSE)

    y_test = tf.keras.utils.to_categorical(y_test, CLASSES_NUM)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Make prediction
    predictions = model.predict(x_test)


def fit_generator_method():
    (x_train, y_train), (x_test, y_test) = process_dataset()

    pivot = int((1-VALIDATION_SPLIT)*len(y_train))
    x_train, x_validation = x_train[:pivot], x_train[pivot:]
    y_train, y_validation = y_train[:pivot], y_train[pivot:]

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

    model.fit(data_generator(x_train, y_train, CLASSES_NUM, BATCH_SIZE),
              steps_per_epoch = x_train.shape[0]/BATCH_SIZE,
              validation_data=data_generator(x_validation, y_validation, CLASSES_NUM, BATCH_SIZE),
              validation_steps = x_validation.shape[0]/BATCH_SIZE,
              epochs=EPOCHS,
              verbose=VERBOSE)

    y_test = tf.keras.utils.to_categorical(y_test, CLASSES_NUM)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Make prediction
    predictions = model.predict(x_test)
if __name__ == "__main__":
    #fit_normal()
    #fit_sequence()
    fit_generator_method()
