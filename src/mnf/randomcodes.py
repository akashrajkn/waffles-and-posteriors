    # kwargs = {
    #     'fq': FLAGS.fq,
    #     'fr': FLAGS.fr,
    #     'no_z': FLAGS.no_z,
    #     'learn_p': FLAGS.learn_p,
    #     'thres_var': FLAGS.thres_var,
    #     'flow_h': FLAGS.flow_h,
    #     'seed': FLAGS.seed,
    #     'anneal': FLAGS.anneal,
    #     'epzero': FLAGS.epzero,
    #     'epochs': FLAGS.epochs,
    #     'lr': FLAGS.lr
    # }
    # graph = build_graph(input_shape, N, kwargs=kwargs)
    # x, y_, y, yd, pyx, regs, cross_entropy, global_step, annealing, lowerbound, train_step, accuracy = graph.get_collection('mnf_stuff')


def build_graph(input_shape, N, kwargs):
    g = tf.Graph()

    with g.as_default():
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

    for node in (x, y_, y, yd, pyx, regs, cross_entropy, global_step, annealing, lowerbound, train_step, accuracy):
        g.add_to_collection("mnf_stuff", node)

    return g
