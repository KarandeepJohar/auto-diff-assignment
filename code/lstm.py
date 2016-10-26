"""
Long Short Term Memory for character level entity classification
"""
import time
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)
EPS = 1e-4

def glorot(m,n):
    # return scale for glorot initialization
    return np.sqrt(6./(m+n))

def grad_check(network):
    # function which takes a network object and checks gradients
    # based on default values of data and params
    dataParamDict = network.graph.inputDict()
    fd = network.fwd(dataParamDict)
    grads = network.bwd(fd)
    for rname in grads:
        if network.graph.isParam(rname):
            fd[rname].ravel()[0] += EPS
            fp = network.fwd(fd)
            a = fp['loss']
            fd[rname].ravel()[0] -= 2*EPS
            fm = network.fwd(fd)
            b = fm['loss']
            fd[rname].ravel()[0] += EPS
            auto = grads[rname].ravel()[0]
            num = (a-b)/(2*EPS)
            if not np.isclose(auto, num, atol=1e-3):
                raise ValueError("gradients not close for %s, Auto %.5f Num %.5f"
                        % (rname, auto, num))

class Network:
    """
    Parent class with functions for doing forward and backward passes through the network, and
    applying updates. All networks should subclass this.
    """
    def display(self):
        print "Operation Sequence:"
        for o in self.graph.operationSequence(self.graph.loss):
            print o

    def fwd(self, valueDict):
        ad = Autograd(self.graph)
        opseq = self.graph.operationSequence(self.graph.loss)

        return ad.eval(opseq, valueDict)

    def bwd(self, valueDict):
        ad = Autograd(self.graph)
        opseq = self.graph.operationSequence(self.graph.loss)
        return ad.bprop(opseq, valueDict,loss=np.float_(1.0))

    def update(self, dataParamDict, grads, rate):
        for rname in grads:
            if self.graph.isParam(rname):
                if grads[rname].shape!=dataParamDict[rname].shape:
                    print rname, grads[rname].shape, dataParamDict[rname].shape
                dataParamDict[rname] = dataParamDict[rname] - rate*grads[rname]
        return dataParamDict

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
        self.build()

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
        x.o1 = f.relu( f.mul(hh,self.params['W1']) + self.params['b1'] )
        x.output = f.softMax(x.o1)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        self.graph = x.setup()
        self.display()

    def data_dict(self, X, y):
        data = {}
        for i in range(self.length):
            data['input_%d'%i] = X[:,i,:]
        data['y'] = y
        data['hid_init'] = np.zeros((X.shape[0],self.num_hid))
        data['cell_init'] = np.zeros((X.shape[0],self.num_hid))
        return data

if __name__=='__main__':
    epochs = 20
    max_len = 10
    num_hid = 50
    batch_size = 16

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('../data/smaller.train', '../data/smaller.test')
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))

    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    print "done"
    # check
    print "checking gradients..."
    grad_check(lstm)
    print "ok"

    # train
    print "training..."
    logger = open('../logs/auto_lstm.txt','w')
    tst = time.time()
    value_dict = lstm.graph.inputDict()
    for i in range(epochs):
        # learning rate schedule
        lr = 0.5/((i+1)**2)

        for (e,l) in mb_train:
            # prepare input
            data_dict = lstm.data_dict(e,l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = lstm.fwd(value_dict)
            gd = lstm.bwd(value_dict)
            value_dict = lstm.update(value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (e,l) in mb_test:
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
        prec = evaluate(np.vstack(probs), np.vstack(targets))

        t_elap = time.time()-tst
        message = 'Epoch %d VAL loss = %.3f prec = %.3f time = %.2f' % (i,tot_loss/n,prec,t_elap)
        logger.write(message+'\n')
        print message
    print "done"
