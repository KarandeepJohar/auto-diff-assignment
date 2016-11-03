"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import time
import numpy as np
from xman import *
from utils import *
from autograd import *
from network import *

np.random.seed(0)
EPS = 1e-4

def glorot(m,n):
    # return scale for glorot initialization
    return np.sqrt(6./(m+n))

class LSTM(Network):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.length = max_len
        self._declareParams()
        self._declareInputs()
        self.my_xman = self.build()
        self.op_seq = self.my_xman.operationSequence(self.my_xman.loss)

    def _declareParams(self):
        scW = glorot(self.in_size,self.num_hid)
        scU = glorot(self.num_hid,self.num_hid)
        scb = 0.1
        self.params = {}
        for suf in ['i','f','o','c']:
            self.params['W'+suf] = f.param(name='W'+suf,
                    default=scW*np.random.uniform(low=-1.,high=1.,
                        size=(self.in_size,self.num_hid)))
            self.params['U'+suf] = f.param(name='U'+suf,
                    default=scU*np.random.uniform(low=-1.,high=1.,
                        size=(self.num_hid,self.num_hid)))
            self.params['b'+suf] = f.param(name='b'+suf,
                    default=scb*np.random.uniform(low=-1.,high=1.,
                        size=(self.num_hid,)))
        sc = glorot(self.num_hid,self.out_size)
        self.params['W1'] = f.param(name='W1', 
                default=sc*np.random.uniform(low=-1.,high=1.,size=(self.num_hid,self.out_size)))
        self.params['b1'] = f.param(name='b1', 
                default=scb*np.random.uniform(low=-1.,high=1.,size=(self.out_size,)))

    def _declareInputs(self):
        self.inputs = {}
        for i in range(self.length):
            self.inputs['input_%d'%i] = f.input(name='input_%d'%i, 
                    default=np.random.rand(1,self.in_size))
        self.inputs['y'] = f.input(name='y', default=np.random.rand(1,self.out_size))
        self.inputs['hid_init'] = f.input(name='hid_init', 
                default=np.zeros((1,self.num_hid)))
        self.inputs['cell_init'] = f.input(name='cell_init', 
                default=np.zeros((1,self.num_hid)))

    @staticmethod
    def _step(t_in, hid, cell, params):
        # one step of lstm computation
        i = f.sigmoid( f.mul(t_in,params['Wi']) + 
                f.mul(hid,params['Ui']) + params['bi'] )
        fg = f.sigmoid( f.mul(t_in,params['Wf']) + 
                f.mul(hid,params['Uf']) + params['bf'] )
        o = f.sigmoid( f.mul(t_in,params['Wo']) + 
                f.mul(hid,params['Uo']) + params['bo'] )
        c_in = f.tanh( f.mul(t_in,params['Wc']) + 
                f.mul(hid,params['Uc']) + params['bc'] )

        cell_n = f.hadamard(fg,cell) + f.hadamard(i,c_in)
        hid_n = f.hadamard(o, f.tanh(cell_n))

        return hid_n, cell_n

    def build(self):
        x = XMan()
        # evaluating the model
        hh = self.inputs['hid_init']
        cc = self.inputs['cell_init']
        for i in range(self.length):
            hh, cc = LSTM._step(self.inputs['input_%d'%i], hh, cc, self.params)
        x.o10 = f.mul(hh,self.params['W1']) + self.params['b1']
        x.o1 = f.relu( x.o10)
        x.output = f.softMax(x.o1)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        return x.setup()

    def data_dict(self, X, y):
        data = {}
        for i in range(self.length):
            data['input_%d'%i] = X[:,-i,:]
        data['y'] = y
        data['hid_init'] = np.zeros((X.shape[0],self.num_hid))
        data['cell_init'] = np.zeros((X.shape[0],self.num_hid))
        return data

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('../data/%s.train'%dataset, '../data/%s.valid'%dataset, '../data/%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    print "done"

    # train
    print "training..."
    # logger = open('../logs/%s_lstm4c_L%d_H%d_B%d_E%d_lr%.3f.txt'%
    #         (dataset,max_len,num_hid,batch_size,epochs,init_lr),'w')
    tst = time.time()
    value_dict = lstm.my_xman.inputDict()
    min_loss = 1e5
    lr = init_lr
    for i in range(epochs):

        for (idxs,e,l) in mb_train:
            # prepare input
            data_dict = lstm.data_dict(e,l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = lstm.fwd(value_dict)
            gd = lstm.bwd(value_dict)
            value_dict = lstm.update(value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            # logger.write(message+'\n')

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (idxs,e,l) in mb_valid:
            # prepare input
            data_dict = lstm.data_dict(e,l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd
            vd = lstm.fwd(value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += 1
        acc = accuracy(np.vstack(probs), np.vstack(targets))
        c_loss = tot_loss/n
        if c_loss<min_loss: min_loss = c_loss

        t_elap = time.time()-tst
        message = ('Epoch %d VAL loss %.3f min_loss %.3f acc %.3f time %.2f' % 
                (i,c_loss,min_loss,acc,t_elap))
        # logger.write(message+'\n')
        print message
    print "done"

    tot_loss, n= 0., 0
    probs = []
    targets = []
    indices = []
    for (idxs,e,l) in mb_test:
        # prepare input
        data_dict = lstm.data_dict(e,l)
        for k,v in data_dict.iteritems():
            value_dict[k] = v
        # fwd
        vd = lstm.fwd(value_dict)
        tot_loss += vd['loss']
        probs.append(vd['output'])
        targets.append(l)
        indices.extend(idxs)
        n += 1
    def newIndices(indices):
        l = [0]*len(indices)
        for i in range(len(indices)):
            l[indices[i]]=i
        return l

    np.save(output_file, np.vstack(probs)[newIndices(indices)])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--dataset', dest='dataset', type=str, default='smaller')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)