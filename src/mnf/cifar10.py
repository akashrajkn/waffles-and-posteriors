import cPickle as pkl
import numpy as np


class CIFAR10(object):
    def __int__(self):
        self.nb_classes = 10
        self.name = self.__class__.__name__.lower()

    def load_data(self):

        with open('../../data/cifar10/data_batch_1', 'rb') as f:
            cifar_training_1 = pkl.load(f)
        with open('../../data/cifar10/data_batch_2', 'rb') as f:
            cifar_training_2 = pkl.load(f)
        with open('../../data/cifar10/data_batch_3', 'rb') as f:
            cifar_training_3 = pkl.load(f)
        with open('../../data/cifar10/data_batch_4', 'rb') as f:
            cifar_training_4 = pkl.load(f)
        with open('../../data/cifar10/data_batch_5', 'rb') as f:
            cifar_training_5 = pkl.load(f)
        with open('../../data/cifar10/test_batch', 'rb') as f:
            cifar_test = pkl.load(f)

        training_data = np.concatenate((cifar_training_1['data'], cifar_training_2['data'], cifar_training_3['data'],
                                        cifar_training_4['data']), axis=0)
        training_labels = np.concatenate((cifar_training_1['labels'], cifar_training_2['labels'], cifar_training_3['labels'],
                                          cifar_training_4['labels']), axis=0)

        # FIXME
        validation_data = cifar_training_5['data'][:1000]
        validation_labels = cifar_training_5['labels'][:1000]

        return [training_data, training_labels], [validation_data, validation_labels], [cifar_test['data'], cifar_test['labels']]

    def permutation_invariant(self, n=None):
        train, valid, test = self.load_data()
        return train, valid, test

    def images(self, n=None):
        train, valid, test = self.load_data()
        train[0] = np.reshape(train[0], (train[0].shape[0], 3, 32, 32))
        valid[0] = np.reshape(valid[0], (valid[0].shape[0], 3, 32, 32))
        test[0] = np.reshape(test[0], (test[0].shape[0], 3, 32, 32))
        return train, valid, test
