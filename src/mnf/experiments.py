import pickle
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from progressbar import ETA, Bar, Percentage, ProgressBar

from mnist import MNIST
from wrappers import MNFLeNet
# from utils import test_mnist_rot, save_mnist_to_file

def experiment_one():

    model_dir = './models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/'

    # pyx = tf.get_variable("pyx")


    # with tf.Session() as sess:
    # sess = tf.InteractiveSession()
    sess = tf.Session()

    mnist = MNIST()
    (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = mnist.images()
    xtrain, xvalid, xtest = np.transpose(xtrain, [0, 2, 3, 1]), np.transpose(xvalid, [0, 2, 3, 1]), np.transpose(xtest, [0, 2, 3, 1])
    ytrain, yvalid, ytest = to_categorical(ytrain, 10), to_categorical(yvalid, 10), to_categorical(ytest, 10)

    N, height, width, n_channels = xtrain.shape
    iter_per_epoch = N / 100

    input_shape = [None, height, width, n_channels]
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')


    model = MNFLeNet(N, input_shape=input_shape, flows_q=2, flows_r=2, use_z=False,
                     learn_p=True, thres_var=0.5, flow_dim_h=50)

    tf.set_random_seed(1)
    np.random.seed(1)
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = tf.nn.softmax(y)



    saver = tf.train.import_meta_graph(model_dir + 'mnf.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


    # saver = tf.train.latest_checkpoint(model_dir + '**mnf**')
    # saver.restore(sess, model_dir + 'mnf.json')
    # saver.restore(sess, model_dir + 'mnf')

    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)

    print("loaded")

    print '------------------------------------------------'
    print '-                MNIST rotated                 -'

    data_path = '../../data/mnist/mnist_rotated.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X = data['X']
        y = data['y']
    else:
        # X, y = test_mnist_rot(plot=False)
        # save_mnist_to_file(X, y)
        pass

    X = X.reshape((X.shape[0], 1, 28, 28))

    print X.shape

    X = np.transpose(X, [0, 2, 3, 1])
    # X = X[:, np.newaxis, :, :]
    y = to_categorical(y, 10)

    print 'Data loaded'

    preds = np.zeros_like(y)
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(10, widgets=widgets)
    pbar.start()
    for i in xrange(10):
        pbar.update(i)
        for j in xrange(1):
            pyxi = sess.run(pyx, feed_dict={x: X[0:10]})
            preds[0:10] += pyxi / 10
    print
    sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(y, 1)))
    print 'Sample test accuracy: {}'.format(sample_accuracy)

    print '------------------------------------------------'
if __name__ == '__main__':
    experiment_one()
