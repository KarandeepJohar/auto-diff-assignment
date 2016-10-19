"""
Implement baselines for character level classification of DBPedia entities.
The implementation uses lasagne and theano. Current models include:
    - MLP
    - LSTM
    - BiLSTM
"""

import time
import argparse
import io
import numpy as np
import theano
import lasagne
import theano.tensor as T
import lasagne.layers as L

from utils import *

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
        self.validate_fn = theano.function(self.inps, [loss,acc,probs])

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
        self.validate_fn = theano.function(self.inps, [loss,acc,probs])

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
                only_return_final=False) # B x N x 50
        l_2 = L.DenseLayer(l_1, 100) # flattens N x D --> ND
        l_out = L.DenseLayer(l_1, self.num_labels, 
                nonlinearity=lasagne.nonlinearities.softmax)
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
        self.validate_fn = theano.function(self.inps, [loss,acc,probs])

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

def evaluate(probs, targets):
    # compute precision @1
    preds = np.argmax(probs, axis=1)
    return float((targets[np.arange(targets.shape[0]),preds]==1).sum())/ \
            targets.shape[0]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, 
            default='MLP', help='Model to use')
    parser.add_argument('--learning_rate', dest='lr', type=float, 
            default=0.01, help='Learning rate')
    params = vars(parser.parse_args())

    dp = DataPreprocessor()
    data = dp.preprocess('../data/small.train','../data/small.test')
    mb_train = MinibatchLoader(data.training, 10, 10, len(data.labeldict))
    mb_test = MinibatchLoader(data.test, 10, 10, len(data.labeldict))

    m = eval(params['model'])(len(data.chardict), 10, 10, len(data.labeldict), params['lr'])

    logger = open('../logs/%s_%.3f_log.txt' % (params['model'].lower(),params['lr']),'w')
    max_prec = 0.
    tst = time.time()
    for epoch in range(100):
        print 'epoch ', epoch
        for (e,l) in mb_train:
            tr_loss, tr_acc = m.train(e,l)
            message = 'TRAIN loss = %.3f acc = %.3f' % (tr_loss, tr_acc)
            logger.write(message + '\n')
            print message
        
        tot_loss, tot_acc = 0., 0.
        n = 0
        probs = []
        targets = []
        for (e,l) in mb_test:
            loss, _, pr = m.validate(e,l)
            probs.append(pr)
            targets.append(l)
            tot_loss += loss
            n += 1
        prec = evaluate(np.vstack(probs), np.vstack(targets))
        if prec>max_prec: max_prec = prec
        message = 'VAL loss = %.3f prec = %.3f max_prec = %.3f' % (tot_loss/n, prec, max_prec)
        logger.write(message + '\n')
        print message
    logger.write('Time elapsed = %.2f\n' % (time.time()-tst))
    logger.close()
