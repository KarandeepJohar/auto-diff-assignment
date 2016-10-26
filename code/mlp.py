"""
Multilayer Perceptron for character level entity classification
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

class MLP(Network):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)-1
        self._declareParams(layer_sizes)
        self._declareInputs(layer_sizes)
        self._build()

    def _declareParams(self, layer_sizes):
        print "INITIAZLIZING with layer_sizes:", layer_sizes
        self.params = {}
        for i in range(self.num_layers):
            k = i+1
            sc = glorot(layer_sizes[i], layer_sizes[i+1])
            self.params['W'+str(k)] = f.param(name='W'+str(k), 
                    default=sc*np.random.uniform(low=-1.,high=1.,
                        size=(layer_sizes[i], layer_sizes[i+1])))
            self.params['b'+str(k)] = f.param(name='b'+str(k), 
                    default=0.1*np.random.uniform(low=-1.,high=1.,size=(layer_sizes[i+1],)))
            
    def _declareInputs(self, layer_sizes):
        # set defaults for gradient check
        self.inputs = {}
        self.inputs['X'] = f.input(name='X', default=np.random.rand(1,layer_sizes[0]))
        self.inputs['y'] = f.input(name='y', default=np.random.rand(1,layer_sizes[-1]))

    def _build(self):
        x = XMan()
        # evaluating the model
        inp = self.inputs['X']
        for i in range(self.num_layers):
            inp = f.relu( f.mul(inp,self.params['W'+str(i+1)]) + self.params['b'+str(i+1)] )
        x.output = f.softMax(inp)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        self.graph = x.setup()
        self.display()

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict

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
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    print "done"
    # check
    print "checking gradients..."
    grad_check(mlp)
    print "ok"

    # train
    print "training..."
    logger = open('../logs/auto_mlp.txt','w')
    tst = time.time()
    value_dict = mlp.graph.inputDict()
    for i in range(epochs):
        # learning rate schedule
        lr = 0.5/((i+1)**2)

        for (e,l) in mb_train:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = mlp.fwd(value_dict)
            gd = mlp.bwd(value_dict)
            value_dict = mlp.update(value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (e,l) in mb_test:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd
            vd = mlp.fwd(value_dict)
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
