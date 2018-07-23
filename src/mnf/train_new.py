import pickle
import random

import tensorflow as tf
import numpy as np

from progressbar import ETA, Bar, Percentage, ProgressBar
from keras.utils.np_utils import to_categorical
from mnist import MNIST
from cifar10 import CIFAR10
import time, os
from wrappers import MNFLeNet
from utils import create_mnist_rot_test_data
from constants import PARAMS


def train(dataset, save_losses):

    kwargs = {
        'fq': 2,
        'fr': 2,
        'no_z': True,
        'learn_p': True,
        'thres_var': 0.5,
        'flow_h': 50,
        'seed': 1,
        'anneal': True,
        'epzero': 1,
        'epochs': 25,
        'lr': 0.001
    }

    prior = PARAMS['prior']

    if dataset == 'mnist':
        mnist = MNIST()
        (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = mnist.images()
    else:
        cifar10 = CIFAR10()
        (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = cifar10.images()

    xtrain, xvalid, xtest = np.transpose(xtrain, [0, 2, 3, 1]), np.transpose(xvalid, [0, 2, 3, 1]), np.transpose(xtest, [0, 2, 3, 1])
    ytrain, yvalid, ytest = to_categorical(ytrain, 10), to_categorical(yvalid, 10), to_categorical(ytest, 10)

    N, height, width, n_channels = xtrain.shape

    if dataset == "mnist":
        iter_per_epoch = N / 100
    else:
        iter_per_epoch = N / 1000

    sess = tf.InteractiveSession()

    input_shape = [None, height, width, n_channels]
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    model = MNFLeNet(N, input_shape=input_shape, flows_q=kwargs['fq'], flows_r=kwargs['fr'], use_z=not kwargs['no_z'],
                     learn_p=kwargs['learn_p'], thres_var=kwargs['thres_var'], flow_dim_h=kwargs['flow_h'])

    tf.set_random_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
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
    if kwargs['anneal']:
        number_zero, original_zero = kwargs['epzero'], kwargs['epochs'] / 2
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

    train_step = tf.train.AdamOptimizer(learning_rate=kwargs['lr']).minimize(lowerbound, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(yd, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(kwargs['logs'] + '/train', sess.graph)

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)

    ################################### Newly Added ##########################################
    # tf.add_to_collection('pyx', pyx)  # This creates some problem
    ################################### Newly Added ##########################################

    saver = tf.train.Saver(tf.global_variables())

    print("---------------------------------------------")
    # print(tf.global_variables())
    print("---------------------------------------------")


    print '{} : TRAIN : {}'.format('_' * 10, '_' * 10)
    print ' - DATASET: {}'.format(dataset)
    print ' - Input layer shape: {}, {}, {}'.format(height, width, n_channels)

    tf.global_variables_initializer().run()

    idx = np.arange(N)
    steps = 0
    model_dir = '../../models/mnf/{}_{}_epochs-{}/'.format(dataset, prior, kwargs['epochs'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print ' - Will save model as: {}'.format(model_dir)

    entropies = []
    lowerbounds = []
    # Train
    for epoch in xrange(kwargs['epochs']):
        widgets = ["epoch {}/{}|".format(epoch + 1, kwargs['epochs']), Percentage(), Bar(), ETA()]
        pbar = ProgressBar(iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        t0 = time.time()

        current_entropy = 0
        current_lowerbound = 0

        for j in xrange(iter_per_epoch):
            steps += 1
            pbar.update(j)

            if dataset == "mnist":
                batch_size = 100
            else:
                batch_size = 10
            batch = np.random.choice(idx, batch_size)
            if j == (iter_per_epoch - 1):
                summary, _, a, b = sess.run([merged, train_step, cross_entropy, lowerbound], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
                # train_writer.add_summary(summary,  steps)
                # train_writer.flush()
            else:
                _, a, b = sess.run([train_step, cross_entropy, lowerbound], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
            current_entropy += a
            current_lowerbound += b

        # print current_entropy
        # print current_lowerbound

        entropies.append(current_entropy / iter_per_epoch)
        lowerbounds.append(current_lowerbound / iter_per_epoch)

        # the accuracy here is calculated by a crude MAP so as to have fast evaluation
        # it is much better if we properly integrate over the parameters by averaging across multiple samples
        tacc = sess.run(accuracy, feed_dict={x: xvalid, y_: yvalid})
        string = '   - Epoch {}/{}, valid_acc: {:0.3f}'.format(epoch + 1, kwargs['epochs'], tacc)

        # if (epoch % 2) == 0:
            # test_1(sess, xtest, ytest, pyx, cross_entropy, lowerbound, x)

        if (epoch + 1) % 10 == 0:
            string += ', model_save: True'
            saver.save(sess, model_dir)

        string += ', dt: {:0.3f}'.format(time.time() - t0)
        print string

    saver.save(sess, model_dir + 'mnf')
    # train_writer.close()

    # *********************** Save cross_entropy and lowerbound *************************
    if save_losses:
        print ' - Save entropies and lowerbound'
        print entropies
        print lowerbounds
        dictionary = {
            'entropies': entropies,
            'lowerbounds': lowerbounds
        }

        with open('../../models/mnf/{}_{}'.format(dataset, prior), 'wb') as dictfile:
            pickle.dump(dictionary, dictfile)
    # ************************************************************************************




def main():

    # ******* Controls *********
    dataset = 'mnist'
    save_losses = True
    # **************************

    train(dataset, save_losses)

if __name__ == '__main__':


    main()
