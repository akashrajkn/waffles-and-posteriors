import random
import tensorflow as tf
import numpy as np

from wrappers import MNFLeNet


def setup_tf_graph(N, input_shape, FLAGS):
    '''
    Setup the tensorflow computation graph
    '''
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    model = MNFLeNet(N, input_shape=input_shape, flows_q=FLAGS['fq'], flows_r=FLAGS['fr'], use_z=not FLAGS['no_z'],
                     learn_p=FLAGS['learn_p'], thres_var=FLAGS['thres_var'], flow_dim_h=FLAGS['flow_h'])

    tf.set_random_seed(FLAGS['seed'])
    np.random.seed(FLAGS['seed'])
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = tf.nn.softmax(y)

    global_step = tf.Variable(0, trainable=False)

    if FLAGS['anneal']:
        number_zero, original_zero = FLAGS['epzero'], FLAGS['epochs'] / 2
        with tf.name_scope('annealing_beta'):
            max_zero_step = number_zero * iter_per_epoch
            original_anneal = original_zero * iter_per_epoch
            beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
            beta_t = tf.maximum(beta_t_val, 0.)
            annealing = tf.minimum(1., tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
            tf.summary.scalar('annealing beta', annealing)
    else:
        annealing = 1.

    with tf.name_scope('KL_prior'):
        regs = model.get_reg()
        tf.summary.scalar('KL prior', regs)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('Loglike', cross_entropy)

    with tf.name_scope('lower_bound'):
        lowerbound = cross_entropy + annealing * regs
        tf.summary.scalar('Lower bound', lowerbound)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(yd, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS['lr']).minimize(lowerbound, global_step=global_step)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train', sess.graph)

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)
    saver = tf.train.Saver(tf.global_variables())

    tf.global_variables_initializer().run()

    return sess, 
