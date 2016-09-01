# This should be completely refactored, shares a lot of code with train.py

import sys
from collections import defaultdict
import numpy as np

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from keras.utils import np_utils

import dataset
import net

np.random.seed(1337)

n = 224
batch_size = 128

data_directory, = sys.argv[1:]

X, y, tags = dataset.dataset(data_directory, n)
nb_classes = len(tags)

sample_count = len(y)
train_size = sample_count * 4 // 5
X_train = X[:train_size]
y_train = y[:train_size]
Y_train = np_utils.to_categorical(y_train, nb_classes)
X_test  = X[train_size:]
y_test  = y[train_size:]
Y_test = np_utils.to_categorical(y_test, nb_classes)


def evaluate(model, vis_filename=None):
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)

    accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    print "accuracy:", accuracy

    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
        confusion[predicted_index, actual_index] += 1
    print "rows are predicted classes, columns are actual classes"
    for predicted_index, predicted_tag in enumerate(tags):
        print predicted_tag[:7],
        for actual_index, actual_tag in enumerate(tags):
            print "\t%d" % confusion[predicted_index, actual_index],
        print
    if vis_filename is not None:
        bucket_size = 10
        image_size = n // 4 # right now that's 56
        vis_image_size = nb_classes * image_size * bucket_size
        vis_image = 255 * np.ones((vis_image_size, vis_image_size, 3), dtype='uint8')
        example_counts = defaultdict(int)
        for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, y_test, X_test):
            example_count = example_counts[(predicted_tag, actual_tag)]
            if example_count >= bucket_size**2:
                continue
            image = dataset.reverse_preprocess_input(normalized_image)
            image = image.transpose((1, 2, 0))
            image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
            tilepos_x = bucket_size * predicted_tag
            tilepos_y = bucket_size * actual_tag
            tilepos_x += example_count % bucket_size
            tilepos_y += example_count // bucket_size
            pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
            vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
            example_counts[(predicted_tag, actual_tag)] += 1
        vis_image[::image_size * bucket_size, :] = 0
        vis_image[:, ::image_size * bucket_size] = 0
        scipy.misc.imsave(vis_filename, vis_image)


model, tags_from_model = net.load("model")
assert tags == tags_from_model
net.compile(model)

evaluate(model, "classifier.png")
