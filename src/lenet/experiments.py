from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

from utils import test_mnist_rot
from lenet import LeNet


def experiment_one():
    X, y = test_mnist_rot(plot=False)

    X = X.reshape((X.shape[0], 28, 28))
    X = X[:, np.newaxis, :, :]
    y = np_utils.to_categorical(y, 10)

    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10,
                        weightsPath='../../models/lenet/lenet_weights.hdf5')
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # randomly select a few testing digits
    for i in np.random.choice(np.arange(0, len(y)), size=(10,)):
        # classify the digit
        probs = model.predict(X[np.newaxis, i])
        prediction = probs.argmax(axis=1)

        # show the image and prediction
        print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
            np.argmax(y[i])))


if __name__ == '__main__':
    experiment_one()
