# Training an InceptionV3-based image classifier with your own dataset

Based on the **Fine-tune InceptionV3 on a new set of classes** example in https://keras.io/applications/

## Dependencies

Very latest (>=1.0.8 from source) Keras, scipy, pillow. opencv2 is only used in the toy webcam app.
See [osx-install.sh](osx-install.sh) for installation instructions on OS X.

## Training

Structure your image files in the following directory hierarchy. Sub-sub directories are
allowed and traversed:

```
data_dir/classname1/*.*
data_dir/classname2/*.*
...
```

It depends on the domain, but a few hundred images per class can already give good results.

Run the training:

```
python train.py data_dir model
```

The standard output provides information about the state of the training, and the current accuracy.
Accuracy is measured on a random 20% validation set. During training, Keras outputs the accuracy on
the augmented validation dataset (`val_acc`). After a training round, the validation accuracy
on non-augmented data is printed.

The files `000.png` `001.png` etc. give a visual confusion matrix about the progress of the training.
`000.png` is created after the newly created dense layers were trained,
and the rest during fine-tuning.

The model is saved in three files, named `model.h5`, `model.json`, `model-labels.json`.

## Webcam integration

If you train the model with the labeled faces of your friends and relatives,
you can test your classifier in a toy app.

```
python webcam.py model
```

This does face detection on the webcam stream, and tags the detected faces according to the neural model.
It looks for the `model*` files in the current directory. The file `haarcascade_frontalface_default.xml`
must also be there.

Webcam data is quite different from photos, so to let the model generalize,
set `heavy_augmentation = True` in `train.py`. For other applications,
`heavy_augmentation = False` might be preferable.


## Apple Photos: a great source of training data for face recognition

OS X Photos users can find high quality training data in the Photos Libraries of that application.
[Mihály Köles](https://github.com/nyuwec) and I have reverse engineered the database format of Photos,
and the result is an easy-to-use tool for building your personalized face recognition
training datasets from Photos Libraries:

```
bash collect-apple-photos.sh "$HOME/Pictures/Photos Library.photoslibrary" photos_library_dataset
```

The output of the above script is the `photos_library_dataset` directory
that has exactly the right layout to be used as input for the training script:

```
python train.py photos_library_dataset model
python webcam.py model
```

Of course, very small label classes won't generalize well to unseen data. It might make sense to
consolidate their contents into the generic `unknown` label class, which contains faces not yet
labeled by Apple Photos:

```
mv photos_library_dataset/too_small_class/* photos_library_dataset/unknown
rmdir photos_library_dataset/too_small_class
```

If you simply remove the `unknown` directory from the dataset before training, that leads to a "closed world"
model that assumes that everyone appearing on your webcam stream has his or her Photos label.


### The Photos Library data layout

For those interested, here's a bit more information about the Photos data layout. (Look into the
source code of `collect-apple-photos.sh` for all relevant detail.)
`*.photoslibrary/resources/modelresources/` contains the manually and semi-automatically tagged
faces cropped from your photos, and the `*.photoslibrary/database/Person.db` and
`*.photoslibrary/database/ImageProxies.apdb` sqlite3 databases describe the correspondence
between persons and cropped photos. The relevant tables for our purposes are
`Person.RKFace`, `Person.RKPerson`, and `ImageProxies.RKModelResource`.
