import cv2
import sys

import numpy as np

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

import net
import dataset


n = 224

print "loading neural network"
model, tags = net.load("model")
net.compile(model)
print "done"

print "compiling predictor function"
_ = model.predict(np.zeros((1, 3, n, n), dtype=np.float32), batch_size=1)
print "done"


cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    frameOut = np.array(frame)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if w>100 and h>100:
            cv2.rectangle(frameOut, (x, y), (x+w, y+h), (0, 255, 0), 2)
            square = frame[max((y-h//2,0)):y+3*h//2, max((x-w//2,0)):x+3*w//2]
            square = scipy.misc.imresize(square.astype(np.float32), size=(n, n), interp='bilinear')
            square = np.expand_dims(square, axis=0)
            square = square.transpose((0, 3, 1, 2))
            square = dataset.preprocess_input(square)

            probabilities = model.predict(square, batch_size=1).flatten()
            prediction = tags[np.argmax(probabilities)]
            print prediction + "\t" + "\t".join(map(lambda x: "%.2f" % x, probabilities))
            cv2.putText(frameOut, prediction, (x, y-2), font, 1, (255,255,255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
