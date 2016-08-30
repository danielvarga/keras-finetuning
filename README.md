# Training an InceptionV3-based image classifier with your own dataset

Based on the **Fine-tune InceptionV3 on a new set of classes** example in https://keras.io/applications/

Structure your data like this:

```
data/classname1/
data/classname2/
```

Run the training:

```
find data -type f | python finetune.py data
```

The standard output will provide information about the state of the training, and the current accuracy.
(Accuracy is measured on a random 20% validation set.)
The files `000.png` `001.png` etc give a visual confusion matrix about the progress of the training.

