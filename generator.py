import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras
    x of `model.fit` could be a generator or keras.utils.Sequence
    returning `(inputs, targets)` or `(inputs, targets, sample_weights)`.

    Every Sequence must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.
    '''

    def __init__(self, x, y, classes_num,
                  batch_size=32, shuffle=True):
        'Initialization'
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.classes_num = classes_num
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = [self.x[idx] for idx in indexes]
        batch_y = [self.y[idx] for idx in indexes]

        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=self.classes_num)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
