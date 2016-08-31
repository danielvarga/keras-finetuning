# Training an InceptionV3-based image classifier with your own dataset

Based on the **Fine-tune InceptionV3 on a new set of classes** example in https://keras.io/applications/

## Dependencies

Very latest (>=1.0.8) Keras, scipy. opencv2 is only used in the toy webcam app.

## Training

Structure your image files in the following directory hierarchy:

```
data/classname1/*.*
data/classname2/*.*
...
```

It depends on the domain, but a few hundred images per class can already give good results.

Run the training:

```
find data -type f | python train.py data
```

The standard output will provide information about the state of the training, and the current accuracy.
(Accuracy is measured on a random 20% validation set.)
The files `000.png` `001.png` etc give a visual confusion matrix about the progress of the training.

The model is saved in three files, named `model.h5`, `model.json`, `model-labels.json`.

## Webcam integration

If you train the model with the labeled faces of your friends and relatives,
you can test your classifier in a toy app.
(OS X Photos users can find training data in the working directory of that application.)

```
python webcam.py haarcascade_frontalface_default.xml
```

This does face detection on the webcam stream, and tags the detected faces according to the neural model.
It looks for the `model*` files in the current directory.
