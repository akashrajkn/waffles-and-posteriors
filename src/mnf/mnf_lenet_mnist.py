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
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

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

    # idx = np.arange(N)
    # steps = 0
    # model_dir = '../../models/mnf/{}_{}_epochs-{}/'.format(dataset, prior, FLAGS.epochs)
    #
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # print ' - Will save model as: {}'.format(model_dir)


    print 'Load weights and apply to network'
    model_path = '../../models/mnf/mnist_{}_epochs-25/'.format(prior)
    saver = tf.train.import_meta_graph('{}mnf.meta'.format(model_path))
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    # entropies = []
    # lowerbounds = []
    # # Train
    # for epoch in xrange(FLAGS.epochs):
    #     widgets = ["epoch {}/{}|".format(epoch + 1, FLAGS.epochs), Percentage(), Bar(), ETA()]
    #     pbar = ProgressBar(iter_per_epoch, widgets=widgets)
    #     pbar.start()
    #     np.random.shuffle(idx)
    #     t0 = time.time()
    #
    #     current_entropy = 0
    #     current_lowerbound = 0
    #
    #     for j in xrange(iter_per_epoch):
    #         steps += 1
    #         pbar.update(j)
    #
    #         if dataset == "mnist":
    #             batch_size = 100
    #         else:
    #             batch_size = 10
    #         batch = np.random.choice(idx, batch_size)
    #         if j == (iter_per_epoch - 1):
    #             summary, _, a, b = sess.run([merged, train_step, cross_entropy, lowerbound], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
    #             train_writer.add_summary(summary,  steps)
    #             train_writer.flush()
    #         else:
    #             _, a, b = sess.run([train_step, cross_entropy, lowerbound], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
    #         current_entropy += a
    #         current_lowerbound += b
    #
    #     # print current_entropy
    #     # print current_lowerbound
    #
    #     entropies.append(current_entropy / iter_per_epoch)
    #     lowerbounds.append(current_lowerbound / iter_per_epoch)
    #
    #     # the accuracy here is calculated by a crude MAP so as to have fast evaluation
    #     # it is much better if we properly integrate over the parameters by averaging across multiple samples
    #     tacc = sess.run(accuracy, feed_dict={x: xvalid, y_: yvalid})
    #     string = '   - Epoch {}/{}, valid_acc: {:0.3f}'.format(epoch + 1, FLAGS.epochs, tacc)
    #
    #     # if (epoch % 2) == 0:
    #         # test_1(sess, xtest, ytest, pyx, cross_entropy, lowerbound, x)
    #
    #     if (epoch + 1) % 10 == 0:
    #         string += ', model_save: True'
    #         saver.save(sess, model_dir)
    #
    #     string += ', dt: {:0.3f}'.format(time.time() - t0)
    #     print string
    #
    # saver.save(sess, model_dir + 'mnf')
    # train_writer.close()
    #
    # # *********************** Save cross_entropy and lowerbound *************************
    # if save_losses:
    #     print ' - Save entropies and lowerbound'
    #     print entropies
    #     print lowerbounds
    #     dictionary = {
    #         'entropies': entropies,
    #         'lowerbounds': lowerbounds
    #     }
    #
    #     with open('../../models/mnf/{}_{}'.format(dataset, prior), 'wb') as dictfile:
    #         pickle.dump(dictionary, dictfile)
    # # ************************************************************************************


    # if not perform_test:
    #     return
    #
    # # ******************************* Test ***********************************************
    #
    # print '***************** EXPERIMENTS *****************'
    # print '>>>>>>>>>>>>>>>>> Prior: {}'.format(prior)
    #
    #
    # # >>>>>>>>>>>>>>>>>>>> Experiment 1: Accuracy
    # print '------------------- Experiment 1: Accuracy -----------------------------'
    # print '-                        {}                      -'.format(dataset)
    #
    # preds = np.zeros_like(ytest)
    # widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    # pbar = ProgressBar(FLAGS.L, widgets=widgets)
    # pbar.start()
    # for i in xrange(FLAGS.L):
    #     pbar.update(i)
    #     for j in xrange(xtest.shape[0] / 100):
    #         pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
    #         preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    #         # print xtest.shape
    #         # print preds.shape
    #         # print FLAGS.L
    #         # print pyxi.shape
    #         # return
    #
    # # save preds
    # print
    # with open('../../models/mnf/{}_{}_preds_experiment_1'.format(dataset, prior, FLAGS.epochs), 'w') as outfile:
    #     np.save(outfile, preds)
    #
    # sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    # print '  - Sample test accuracy: {}'.format(sample_accuracy)
    #
    #
    # >>>>>>>>>>>>>>>>>>>> Experiment 2: Compressed weights on the dense layer
    threshold_dense_layer_1 = 0.5
    threshold_dense_layer_2 = 0.1

    print '------------------- Experiment 2: Compressed weights -----------------------------'

    # to save:
    compression_accuracy = 'threshold, compression, accuracy\n'

    # compression in layer 1
    cutoffs = np.linspace(0.0, 1.0, num=10)
    # Test code:
    # cutoffs = [0.0, 0.2]
    weights = [v for v in tf.trainable_variables() if v.name == "fq2_fr2_usezTrue/densemnf_1/mean_W:0"][0]
    total_non_zeros = np.count_nonzero(weights.eval())
    # total_non_zeros_tf = tf.count_nonzero(weights)
    # print 'np = {}, tf = {}'.format(total_non_zeros, total_non_zeros_tf)

    for cutoff in cutoffs:
        temp = weights.eval()
        temp[(temp >= -cutoff) & (temp <= cutoff)] = 0.0

        w_new = [v for v in tf.trainable_variables() if v.name == "fq2_fr2_usezTrue/densemnf_1/mean_W:0"][0]
        w_new.load(temp.tolist(), sess)
        sess.run(w_new)

        print "******** {} ********".format(cutoff)
        preds = np.zeros_like(ytest)
        widgets = ["Sampling |", Percentage(), Bar(), ETA()]
        pbar = ProgressBar(FLAGS.L, widgets=widgets)
        pbar.start()
        for i in xrange(FLAGS.L):
            pbar.update(i)
            for j in xrange(xtest.shape[0] / 100):
                pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
                preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
        print
        sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
        print '  - Sample test accuracy: {}'.format(sample_accuracy)

        current_non_zeros = np.count_nonzero(temp) * 1.0
        # current_non_zeros = tf.count_nonzero(w_new)

        print 'total = {}, current = {}'.format(total_non_zeros, current_non_zeros)
        compression_current   = (total_non_zeros - current_non_zeros) / total_non_zeros
        compression_accuracy += '{}, {}, {}\n'.format(cutoff, compression_current, sample_accuracy)

    # save compression to file
    with open('../../results/compression_accuracy_{}_layer1.csv'.format(prior), 'w+') as f:
        f.write(compression_accuracy)

    # weights = [v for v in tf.trainable_variables() if v.name == "fq2_fr2_usezTrue/densemnf_2/mean_W:0"][0]
    # weights_eval = weights.eval()
    # weights_eval[(weights_eval >= -threshold_dense_layer_1) & (weights_eval <= threshold_dense_layer_1)] = 0.0
    # w_new = [v for v in tf.trainable_variables() if v.name == "fq2_fr2_usezTrue/densemnf_2/mean_W:0"][0]
    # w_new.load(weights_eval.tolist(), sess)
    # sess.run(w_new)
    #
    # preds = np.zeros_like(ytest)
    # widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    # pbar = ProgressBar(FLAGS.L, widgets=widgets)
    # pbar.start()
    # for i in xrange(FLAGS.L):
    #     pbar.update(i)
    #     for j in xrange(xtest.shape[0] / 100):
    #         pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
    #         preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    #
    # # # save preds
    # # with open('../../models/mnf/{}_{}_preds_experiment_2'.format(dataset, prior, FLAGS.epochs), 'w') as outfile:
    # #     np.save(outfile, preds)
    #
    # print
    # sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    # print '  - Sample test accuracy: {}'.format(sample_accuracy)

    #
    # # >>>>>>>>>>>>>>>>>>>> Experiment 3: Rotated MNIST
    #
    # if dataset != 'mnist':
    #     return
    #
    # print '------------------- Experiment 3: Rotated MNIST -----------------------------'
    #
    # test_data_path = '../../data/mnist/mnist_rotation_test.pkl'
    # if not os.path.exists(test_data_path):
    #     data_test = create_mnist_rot_test_data()
    # else:
    #     with open(test_data_path, 'rb') as f:
    #         data_test = pickle.load(f)
    #
    # X = data_test['xtest']
    # y = data_test['ytest']
    #
    # X = X.reshape((X.shape[0], 1, 28, 28))
    # X = np.transpose(X, [0, 2, 3, 1])
    # y = to_categorical(y, 10)
    #
    # print '  - Data loaded'
    #
    # preds = np.zeros_like(y)
    # widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    # pbar = ProgressBar(FLAGS.L, widgets=widgets)
    # pbar.start()
    # for i in xrange(FLAGS.L):
    #     pbar.update(i)
    #     for j in xrange(X.shape[0] / 100):
    #         pyxi = sess.run(pyx, feed_dict={x: X[j * 100:(j + 1) * 100]})
    #         preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    #
    # # save preds
    # with open('../../models/mnf/{}_{}_preds_experiment_3'.format(dataset, prior, FLAGS.epochs), 'w') as outfile:
    #     np.save(outfile, preds)
    #
    # print
    # sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(y, 1)))
    # print '  - Sample test accuracy: {}'.format(sample_accuracy)
    #
    # # >>>>>>>>>>>>>>>>>>>> Experiment 4: not MNIST
    #
    # if dataset != 'mnist':
    #     return
    #
    # print '------------------- Experiment 4: not MNIST -----------------------------'
    #
    # test_data_path = '../../data/mnist/notmnist_test.pkl'
    # with open(test_data_path, 'rb') as f:
    #     data_test = pickle.load(f)
    #
    # X = data_test['xtest']
    # y = data_test['ytest']
    #
    # X = X.reshape((X.shape[0], 1, 28, 28))
    # X = np.transpose(X, [0, 2, 3, 1])
    # y = to_categorical(y, 10)
    #
    # print '  - Data loaded'
    #
    # preds = np.zeros_like(y)
    # widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    # pbar = ProgressBar(FLAGS.L, widgets=widgets)
    # pbar.start()
    # for i in xrange(FLAGS.L):
    #     pbar.update(i)
    #     for j in xrange(X.shape[0] / 100):
    #         pyxi = sess.run(pyx, feed_dict={x: X[j * 100:(j + 1) * 100]})
    #         preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    #
    # # save preds
    # with open('../../models/mnf/{}_{}_preds_experiment_4'.format(dataset, prior, FLAGS.epochs), 'w') as outfile:
    #     np.save(outfile, preds)
    #
    # print
    # sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(y, 1)))
    # print '  - Sample test accuracy: {}'.format(sample_accuracy)
    #
    # print '------------------------------------------------'
    #
    # print '------------------- Experiment 5: rot90 MNIST -----------------------------'
    #
    # test_data_path = '../../data/mnist/mnist_test_rot90.pkl'
    # with open(test_data_path, 'rb') as f:
    #     data_test = pickle.load(f)
    #
    # X = data_test['xtest']
    # y = data_test['ytest']
    #
    # X = X.reshape((X.shape[0], 1, 28, 28))
    # X = np.transpose(X, [0, 2, 3, 1])
    #
    # preds = np.zeros_like(y)
    # widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    # pbar = ProgressBar(FLAGS.L, widgets=widgets)
    # pbar.start()
    # for i in xrange(FLAGS.L):
    #     pbar.update(i)
    #     for j in xrange(X.shape[0] / 100):
    #         pyxi = sess.run(pyx, feed_dict={x: X[j * 100:(j + 1) * 100]})
    #         # print X.shape
    #         # print FLAGS.L
    #         # print pyxi.shape
    #         preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    #
    # # save preds
    # with open('../../models/mnf/{}_{}_preds_experiment_5'.format(dataset, prior, FLAGS.epochs), 'w') as outfile:
    #     np.save(outfile, preds)
    #
    # print
    # sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(y, 1)))
    # print '  - Sample test accuracy: {}'.format(sample_accuracy)
    #
    # print '------------------------------------------------'


def prior_global():
    global PRIOR
    PRIOR = FLAGS.prior

def main():
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # prior_global()

    # ******* Controls *********
    dataset = 'mnist'
    save_losses = True
    # **************************

    train(dataset, save_losses)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='logs/mnf_lenet',
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=10)
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
    # parser.add_argument('-prior', type=str, default='standard_normal')
    FLAGS = parser.parse_args()

    main()
