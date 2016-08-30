import sys

import numpy as np
from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import np_utils

import dataset


np.random.seed(1337)

n = 224
batch_size = 128
nb_epoch = 10
nb_phase_two_epoch = 10

data_directory, = sys.argv[1:]

X, y, tags = dataset.dataset(sys.stdin, data_directory, n)
nb_classes = len(tags)


sample_count = len(y)
train_size = sample_count * 4 // 5
X_train = X[:train_size]
y_train = y[:train_size]
Y_train = np_utils.to_categorical(y_train, nb_classes)
X_test  = X[train_size:]
y_test  = y[train_size:]
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True,
    vertical_flip=False)
datagen.fit(X_train)

test_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=False,
    vertical_flip=False)
test_datagen.fit(X_test)


def evaluate(model, vis_filename=None):
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
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


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])


# train the model on the new data for a few epochs

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch,
            validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size),
            nb_val_samples=X_test.shape[0],
            )


evaluate(model, "pretrain.png")


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

for i in range(10):
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_phase_two_epoch,
            validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size),
            nb_val_samples=X_test.shape[0],
            )

    evaluate(model, str(i).zfill(3)+".png")
