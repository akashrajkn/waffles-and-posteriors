import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers import SGD
from keras.utils import np_utils

from utils import test_mnist_rot, save_mnist_to_file
from lenet import LeNet


def experiment_one(test_set='mnist', plot=False):

    test_data_path = '../../data/mnist/{}_test.pkl'.format(test_set)
    with open(test_data_path, 'rb') as f:
        data_test = pickle.load(f)

    # data_path = '../../data/mnist/mnist_rotated.pkl'
    # if os.path.exists(data_path):
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #     X = data['X']
    #     y = data['y']
    # else:
    #     X, y = test_mnist_rot(plot=False)
    #     save_mnist_to_file(X, y)

    X = data_test['xtest']
    y = data_test['ytest']

    X = X.reshape((X.shape[0], 28, 28))
    X = X[:, np.newaxis, :, :]
    y = np_utils.to_categorical(y, 10)

    # X = X.reshape((X.shape[0], 1, 28, 28))
    # X = np.transpose(X, [0, 2, 3, 1])
    # y = to_categorical(y, 10)

    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10, weightsPath='../../models/lenet/lenet_weights.hdf5')
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    probs_dict = {}
    probs_x = []
    predicted = []
    predicted_nums = []
    preds = np.zeros_like(y)

    for i in range(10000): #np.random.choice(np.arange(0, len(y)), size=(1,)):
        # classify the digit
        probs = model.predict(X[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        preds[i] = probs
        # print(probs)

        for j in range(10):
            current_probs = probs_dict.get(j)
            if current_probs is None:
                probs_dict[j] = [probs[0][j]]
            else:
                probs_dict[j] = current_probs + [probs[0][j]]

        probs_x.append(np.argmax(y[i]))
        predicted.append(probs.max(axis=1))
        predicted_nums.append(probs.argmax(axis=1))

        # print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(y[i])))

    if test_set == 'notmnist':
        exp_num = 4
    elif test_set == 'mnist_rotation':
        exp_num = 3

    with open('../../models/lenet/preds_experiment_{}_{}'.format(str(exp_num), test_set), 'w') as outfile:
        np.save(outfile, preds)

    if plot:
        for_legend = []
        x_axis = np.arange(10)
        plt.figure()

        for i in range(10):
            for_legend.append(plt.scatter(x_axis, probs_dict[i], marker='.'))

        # plt.scatter(x_axis, predicted, s=80, facecolors='none', edgecolors='r')

        for i in range(10):
            plt.annotate(str(predicted_nums[i][0]), (x_axis[i], predicted[i]))

        plt.xticks(x_axis, probs_x)
        plt.ylabel('Softmax probability')
        plt.legend(for_legend, x_axis, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.show()


if __name__ == '__main__':
    experiment_one(test_set='notmnist', plot=False)
