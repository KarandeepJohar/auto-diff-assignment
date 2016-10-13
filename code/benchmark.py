"""
Implement baselines for character level classification of DBPedia entities.
The implementation uses lasagne and theano. Current models include:
    - MLP
    - LSTM
"""

import argparse
import io
import numpy as np
import theano
import lasagne
import theano.tensor as T
import lasagne.layers as L

class Data:
    def __init__(self, training, test, chardict, labeldict):
        self.chardict = chardict
        self.labeldict = labeldict
        self.training = training
        self.test = test

class DataPreprocessor:
    def preprocess(self, train_file, test_file):
        """
        preprocess train and test files into one Data object.
        construct character dict from both
        """
        chardict, labeldict = self.make_dictionary(train_file, test_file)
        print 'preparing training data'
        training = self.parse_file(train_file, chardict, labeldict)
        print 'preparing test data'
        test = self.parse_file(test_file, chardict, labeldict)

        return Data(training, test, chardict, labeldict)

    def make_dictionary(self, train_file, test_file):
        """
        go through train and test data and get character and label vocabulary
        """
        print 'constructing vocabulary'
        train_set, test_set = set(), set()
        label_set = set()
        ftrain = io.open(train_file, 'r')
        for line in ftrain:
            entity, label = line.rstrip().split('\t')[:2]
            train_set |= set(list(entity))
            label_set |= set(label.split(','))
        ftest = io.open(test_file, 'r')
        for line in ftest:
            entity, label = line.rstrip().split('\t')[:2]
            test_set |= set(list(entity))
            label_set |= set(label.split(','))
        
        print '# chars in training ', len(train_set)
        print '# chars in testing ', len(test_set)
        print '# chars in (testing-training) ', len(test_set-train_set)
        print '# labels', len(label_set)

        vocabulary = list(train_set | test_set)
        vocab_size = len(vocabulary)
        chardict = dict(zip(vocabulary, range(1,vocab_size+1)))
        chardict[u' '] = 0
        labeldict = dict(zip(list(label_set), range(len(label_set))))
        
        return chardict, labeldict

    def parse_file(self, infile, chardict, labeldict):
        """
        get all examples from a file. 
        replace characters and labels with their lookup
        """
        examples = []
        fin = io.open(infile, 'r')
        for line in fin:
            entity, label = line.rstrip().split('\t')[:2]
            ent = map(lambda c:chardict[c], list(entity))
            lab = map(lambda l:labeldict[l], label.split(','))
            examples.append((ent, lab))
        fin.close()
        return examples

class MinibatchLoader:
    def __init__(self, examples, batch_size, max_len, num_labels):
        self.batch_size = batch_size
        self.max_len = max_len
        self.examples = examples
        self.num_examples = len(examples)
        self.num_labels = num_labels
        self.reset()

    def __iter__(self):
        """ make iterable """
        return self

    def reset(self):
        """ next epoch """
        self.permutation = np.random.permutation(self.num_examples)
        self.ptr = 0

    def next(self):
        """ get next batch of examples """
        if self.ptr>self.num_examples-self.batch_size:
            self.reset()
            raise StopIteration()

        ixs = range(self.ptr,self.ptr+self.batch_size)
        self.ptr += self.batch_size

        e = np.zeros((self.batch_size, self.max_len), dtype='int32') # entity
        l = np.zeros((self.batch_size, self.num_labels), dtype='int32') # labels
        for n, ix in enumerate(ixs):
            ent, lab = self.examples[self.permutation[ix]]
            e[n,:min(len(ent),self.max_len)] = np.array(ent[:self.max_len])
            l[n,lab] = 1

        return e, l

class MLP:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars*max_len
        self.batch_size = batch_size
        self.num_labels = num_labels

        # build network
        ent_var = T.wmatrix('ent')
        lab_var = T.imatrix('lab')
        self.inps = [ent_var, lab_var]
        probs, network = self.build_network()
        params = L.get_all_params(network, trainable=True)

        # loss
        loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        updates = lasagne.updates.sgd(loss, params, learning_rate)
        acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        self.validate_fn = theano.function(self.inps, [loss,acc])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1]*self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii*self.num_chars+idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def validate(self, e, l):
        return self.validate_fn(self.to1hot(e), l)

    def build_network(self):
        l_in = L.InputLayer(shape=(self.batch_size, self.in_size), input_var=self.inps[0])
        l_1 = L.DenseLayer(l_in, 100)
        l_2 = L.DenseLayer(l_1, 20)
        l_out = L.DenseLayer(l_2, self.num_labels, nonlinearity=lasagne.nonlinearities.softmax)
        p = L.get_output(l_out)
        return p, l_out

class LSTM:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars
        self.batch_size = batch_size
        self.num_labels = num_labels

        # build network
        ent_var = T.wtensor3('ent')
        lab_var = T.imatrix('lab')
        self.inps = [ent_var, lab_var]
        probs, network = self.build_network()
        params = L.get_all_params(network, trainable=True)

        # loss
        loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        updates = lasagne.updates.sgd(loss, params, learning_rate)
        acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        self.validate_fn = theano.function(self.inps, [loss,acc])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1],self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii,idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def validate(self, e, l):
        return self.validate_fn(self.to1hot(e), l)

    def build_network(self):
        l_in = L.InputLayer(shape=(self.batch_size, self.max_len, self.in_size), 
                input_var=self.inps[0])
        l_1 = L.LSTMLayer(l_in, 50, backwards=False,
                only_return_final=True) # B x 100
        l_out = L.DenseLayer(l_1, self.num_labels, nonlinearity=lasagne.nonlinearities.softmax)
        p = L.get_output(l_out)
        return p, l_out

class BiLSTM:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars
        self.batch_size = batch_size
        self.num_labels = num_labels

        # build network
        ent_var = T.wtensor3('ent')
        lab_var = T.imatrix('lab')
        self.inps = [ent_var, lab_var]
        probs, network = self.build_network()
        params = L.get_all_params(network, trainable=True)

        # loss
        loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        updates = lasagne.updates.sgd(loss, params, learning_rate)
        acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        self.validate_fn = theano.function(self.inps, [loss,acc])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1],self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii,idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def validate(self, e, l):
        return self.validate_fn(self.to1hot(e), l)

    def build_network(self):
        l_in = L.InputLayer(shape=(self.batch_size, self.max_len, self.in_size), 
                input_var=self.inps[0])
        l_1 = L.LSTMLayer(l_in, 50, backwards=False,
                only_return_final=True) # B x 100
        l_2 = L.LSTMLayer(l_in, 50, backwards=True,
                only_return_final=True) # B x 100
        l_h = L.ConcatLayer([l_1,l_2])
        l_out = L.DenseLayer(l_h, self.num_labels, nonlinearity=lasagne.nonlinearities.softmax)
        p = L.get_output(l_out)
        return p, l_out

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, 
            default='MLP', help='Model to use')
    params = vars(parser.parse_args())

    dp = DataPreprocessor()
    data = dp.preprocess('../data/small.train','../data/small.test')
    mb_train = MinibatchLoader(data.training, 10, 10, len(data.labeldict))
    mb_test = MinibatchLoader(data.test, 10, 10, len(data.labeldict))

    m = eval(params['model'])(len(data.chardict), 10, 10, len(data.labeldict), 0.01)

    logger = open('../logs/%s_log.txt' % params['model'].lower(),'w')
    for epoch in range(20):
        print 'epoch ', epoch
        for (e,l) in mb_train:
            tr_loss, tr_acc = m.train(e,l)
            message = 'TRAIN loss = %.3f acc = %.3f' % (tr_loss, tr_acc)
            logger.write(message + '\n')
            print message
        
        tot_loss, tot_acc = 0., 0.
        n = 0
        for (e,l) in mb_test:
            loss, acc = m.validate(e,l)
            tot_loss += loss
            tot_acc += acc
            n += 1
        message = 'VAL loss = %.3f acc = %.3f' % (tot_loss/n, tot_acc/n)
        logger.write(message + '\n')
        print message
    logger.close()
