import time, os
import pickle
import random

import tensorflow as tf
import numpy as np

from progressbar import ETA, Bar, Percentage, ProgressBar
from keras.utils.np_utils import to_categorical
from mnist import MNIST
from cifar10 import CIFAR10

from wrappers import MNFLeNet
from utils import create_mnist_rot_test_data
from constants import PARAMS


def build_graph(dataset, N, input_shape, prior, experiment, save=False):

    # sess = tf.InteractiveSession()
    tf.reset_default_graph()
    # new_graph = tf.Graph()
    # with tf.Session(graph=new_graph) as sess:

    sess = tf.InteractiveSession()

    iter_per_epoch = N / 100
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    model = MNFLeNet(N, input_shape=input_shape, flows_q=FLAGS.fq, flows_r=FLAGS.fr, use_z=not FLAGS.no_z,
                     learn_p=FLAGS.learn_p, thres_var=FLAGS.thres_var, flow_dim_h=FLAGS.flow_h)

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = tf.nn.softmax(y)

    with tf.name_scope('KL_prior'):
        regs = model.get_reg()
        tf.summary.scalar('KL prior', regs)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('Loglike', cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    if FLAGS.anneal:
        number_zero, original_zero = FLAGS.epzero, FLAGS.epochs / 2
        with tf.name_scope('annealing_beta'):
            max_zero_step = number_zero * iter_per_epoch
            original_anneal = original_zero * iter_per_epoch
            beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
            beta_t = tf.maximum(beta_t_val, 0.)
            annealing = tf.minimum(1., tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
            tf.summary.scalar('annealing beta', annealing)
    else:
        annealing = 1.

    with tf.name_scope('lower_bound'):
        lowerbound = cross_entropy + annealing * regs
        tf.summary.scalar('Lower bound', lowerbound)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(lowerbound, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(yd, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)

    saver = tf.train.Saver(tf.global_variables())
    tf.global_variables_initializer().run()

    print ' ** Prior: {} **'.format(prior)

    model_path = '../../models/mnf/{}_{}_epochs-{}/'.format(dataset, prior, FLAGS.epochs)
    saver = tf.train.import_meta_graph('{}mnf.meta'.format(model_path))
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print ' - Session restored'

    return sess, pyx, x


def test(sess, pyx, x, dataset, xtest, ytest, N, input_shape, prior, experiment, save=False):

    print '{} EXPERIMENT: {} {}'.format('-' * 10, experiment, '-' * 10)

    preds = np.zeros_like(ytest)
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.L, widgets=widgets)
    pbar.start()

    for i in xrange(FLAGS.L):
        pbar.update(i)
        for j in xrange(xtest.shape[0] / 100):
            pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
            preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L

    if save:
        print ' - Save predictions'
        with open('../../models/mnf/preds_{}_{}'.format(prior, experiment), 'w') as outfile:
            np.save(outfile, preds)

    sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    print '  - Sample test accuracy: {}'.format(sample_accuracy)

    print '-' * 30


def prepare_test_data(test_data_path, reshape_x=True, reshape_y=True):
    '''
    Just a helper function to reshape test data in the required format
    '''
    with open(test_data_path, 'rb') as f:
        data_test = pickle.load(f)

    X = data_test['xtest']
    y = data_test['ytest']

    if reshape_x:
        X = X.reshape((X.shape[0], 1, 28, 28))
        X = np.transpose(X, [0, 2, 3, 1])

    if reshape_y:
        y = to_categorical(y, 10)

    return X, y


def get_input_shape():

    mnist = MNIST()
    (xtrain, ytrain), _, _ = mnist.images()
    xtrain = np.transpose(xtrain, [0, 2, 3, 1])
    ytrain = to_categorical(ytrain, 10)

    N, height, width, n_channels = xtrain.shape
    input_shape = [None, height, width, n_channels]

    return N, input_shape


def main():
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # MNIST Experiments
    prior = PARAMS['prior']
    N, input_shape = get_input_shape()
    save = True

    sess, pyx, x = build_graph('mnist', N, input_shape, prior, '3_MNIST-MNIST-rot', save)

    # >>>>>>>>>>>>>>>>>>>>> EXPERIMENT 1
    mnist = MNIST()
    _, _, (xtest, ytest) = mnist.images()
    xtest = np.transpose(xtest, [0, 2, 3, 1])
    ytest = to_categorical(ytest, 10)

    test(sess, pyx, x, 'mnist', xtest, ytest, N, input_shape, prior, '1_MNIST-MNIST', save)

    # >>>>>>>>>>>>>>>>>>>>> EXPERIMENT 2
    # Compressed weights


    # >>>>>>>>>>>>>>>>>>>>> EXPERIMENT 3
    xtest, ytest = prepare_test_data('../../data/mnist/mnist_rotation_test.pkl')
    test(sess, pyx, x, 'mnist', xtest, ytest, N, input_shape, prior, '3_MNIST-MNIST-rot', save)


    # >>>>>>>>>>>>>>>>>>>>> EXPERIMENT 4
    xtest, ytest = prepare_test_data('../../data/mnist/notmnist_test.pkl')
    test(sess, pyx, x, 'mnist', xtest, ytest, N, input_shape, prior, '4_MNIST-notMNIST', save)


    # >>>>>>>>>>>>>>>>>>>>> EXPERIMENT 5
    xtest, ytest = prepare_test_data('../../data/mnist/mnist_test_rot90.pkl', reshape_y=False)
    test(sess, pyx, x, 'mnist', xtest, ytest, N, input_shape, prior, '5_MNIST-MNIST-rot90', save)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='logs/mnf_lenet',
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=25)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-fq', default=2, type=int)
    parser.add_argument('-fr', default=2, type=int)
    parser.add_argument('-no_z', action='store_true')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-thres_var', type=float, default=0.5)
    parser.add_argument('-flow_h', type=int, default=50)
    parser.add_argument('-L', type=int, default=10)
    parser.add_argument('-anneal', action='store_true')
    parser.add_argument('-learn_p', action='store_true')
    FLAGS = parser.parse_args()

    main()
