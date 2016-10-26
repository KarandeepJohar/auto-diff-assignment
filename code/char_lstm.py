# a sample use of xman for learning tasks
# 

import time
import sys
import numpy as np
import scipy.stats as ss
from xman import *
import struct
from utils import *
from autograd import *

np.random.seed(0)
EPS = 1e-4
#
# some test cases
#

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
            print '%s Auto %.5f Num %.5f' % (rname, auto, num)
            if not np.isclose(auto, num, atol=1e-3):
                raise ValueError("gradients not close for %s, Auto %.5f Num %.5f"
                        % (rname, auto, num))


class Network:
    """
    Parent class with functions for doing forward and backward passes through the network, and
    applying updates. All networks should subclass this.
    """
    def display(self):
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

    def __init__(self, L, in_size, num_hid, out_size):
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.length = L
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

def scale(m):
    return 0.01

def glorot(m,n):
    return np.sqrt(12./(m+n))


class MLP(Network):

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
            self.params['W'+str(k)] = f.param(name='W'+str(k), default=sc*np.random.rand(layer_sizes[i], layer_sizes[i+1]))
            self.params['b'+str(k)] = f.param(name='b'+str(k), default=sc*np.random.rand(layer_sizes[i+1]))
            
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

    def init_params(self, in_size, out_size):
        hid1, hid2 = 100, 20
        paramDict = {}
        paramDict['W1'] = 0.1*np.random.rand(in_size,hid1)
        paramDict['b1'] = 0.1*np.random.rand(hid1)
        paramDict['W2'] = 0.01*np.random.rand(hid1,hid2)
        paramDict['b2'] = 0.01*np.random.rand(hid2)
        paramDict['W3'] = 0.05*np.random.rand(hid2,out_size)
        paramDict['b3'] = 0.05*np.random.rand(out_size)

        return paramDict

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

def learn(net, dataDict, initDict, epochs=10, rate=1.0, batch_size=100):
    def dvals(d,keys):
        return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
    x,y = dataDict['input'], dataDict['y']
    dataParamDict = net.graph.inputDict(**initDict)
    for i in range(epochs):
        for j in range(0,x.shape[0],batch_size):
            x_c = x[j:j+batch_size,:]
            y_c = y[j:j+batch_size,:]
            #dataParamDict['input_0'] = x_c
            #dataParamDict['y'] = y_c
            #dataParamDict['hid_init'] = np.zeros((100,5))
            #dataParamDict['cell_init'] = np.zeros((100,5))
            dataParamDict['X'] = x_c
            dataParamDict['y'] = y_c
            vd = net.fwd(dataParamDict)
            gd = net.bwd(dataParamDict)
            dataParamDict = net.update(dataParamDict, gd, rate)
        print 'epoch:',i+1,dvals(vd,'loss'), error(vd['y'], vd['output']) 
        if vd['loss'] < 0.001:
            print 'good enough'
            break
    return dataParamDict

def bhuwanMLP(x,y):
    mlp = MLP()
    h = mlp.graph
    ad = Autograd(h)
    init_dict = mlp.init_params(x.shape[1],y.shape[1])

    dataDict = {
            'input': x,
            'y': y,
            }
    fwd = learn(mlp, dataDict, init_dict, epochs=50, rate=0.1, batch_size=100)
    fwd['X'] = x
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(**fwd))
    print 'learned/target predictions, MLP'
    print np.hstack([fpd['output'], y])
    print 'learned weights, biases'
    print np.vstack([fpd['W1'], fpd['b1']])
    print np.vstack([fpd['W2'], fpd['b2']])
    print fpd['W1'].shape, fpd['b1'].shape
    print fpd['W2'].shape, fpd['b2'].shape

def bhuwanLSTM(x,y):
    lstm = LSTM(1)
    h = lstm.graph
    ad = Autograd(h)
    init_dict = lstm.init_params(x.shape[1],5,y.shape[1])

    dataDict = {
            'input': x,
            'y': y,
            }
    fwd = learn(lstm, dataDict, init_dict, epochs=50, rate=1, batch_size=100)
    fwd['input_0'] = x
    fwd['input_1'] = x
    fwd['input_2'] = x
    fwd['y'] = y
    fwd['hid_init'] = np.zeros((x.shape[0],5))
    fwd['cell_init'] = np.zeros((x.shape[0],5))
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(**fwd))
    print 'learned/target predictions, MLP'
    print fpd['output'].shape
    print y.shape
    print np.hstack([fpd['output'], y])
    print 'learned weights, biases'
    print np.vstack([fpd['W1'], fpd['b1']])
    print np.vstack([fpd['W2'], fpd['b2']])
    print fpd['W1'].shape, fpd['b1'].shape
    print fpd['W2'].shape, fpd['b2'].shape





def entityMLP(train, test, num_chars, max_len, num_hid):
    epochs = 20

    print "building mlp..."
    mlp = MLP([max_len*num_chars, num_hid, train.num_labels])
    print "done"
    # grad check
    paramDict = mlp.graph.inputDict()
    print paramDict['W1']
    grad_check(mlp, paramDict)
    #params = mlp.init_params(2,2)
    #data = mlp.data_dict(np.random.rand(5,2),np.random.rand(5,2))
    #_grad_check(mlp, data, params)
    dataParamDict = mlp.graph.inputDict()
    print "training..."
    logger = open('../logs/auto_mlp.txt','w')
    tst = time.time()
    for i in range(epochs):
        lr = 0.5/((i+1)**2)
        for (e,l) in train:
            # prepare input
            dataParamDict['X'] = e.reshape((e.shape[0],e.shape[1]*e.shape[2]))
            dataParamDict['y'] = l
            # fwd-bwd
            vd = mlp.fwd(dataParamDict)
            gd = mlp.bwd(dataParamDict)
            dataParamDict = mlp.update(dataParamDict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')
            #print message

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (e,l) in test:
            # prepare input
            dataParamDict['X'] = e.reshape((e.shape[0],e.shape[1]*e.shape[2]))
            dataParamDict['y'] = l
            # fwd
            vd = mlp.fwd(dataParamDict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += 1
        prec = evaluate(np.vstack(probs), np.vstack(targets))

        t_elap = time.time()-tst
        message = 'Epoch %d VAL loss = %.3f prec = %.3f time = %.2f' % (i,tot_loss/n,prec,t_elap)
        logger.write(message+'\n')
        print message

def entityLSTM(train, test, num_chars, max_len, num_hid):
    epochs = 20

    print "building lstm..."
    lstm = LSTM(max_len, num_chars, num_hid, train.num_labels)
    print "done"
    print "checking gradients..."
    grad_check(lstm)
    print "ok"
    value_dict = lstm.graph.inputDict()
    print "training..."
    logger = open('../logs/auto_lstm.txt','w')
    tst = time.time()
    for i in range(epochs):
        # learning rate schedule
        lr = 0.5/((i+1)**2)

        for (e,l) in train:
            # prepare input
            data_dict = lstm.data_dict(e,l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = lstm.fwd(value_dict)
            gd = lstm.bwd(value_dict)
            value_dict = lstm.update(value_dict, gd, 0.1)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (e,l) in test:
            # prepare input
            data_dict = lstm.data_dict(e,l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd
            vd = lstm.fwd(value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += e.shape[0]
        prec = evaluate(np.vstack(probs), np.vstack(targets))
        message = 'VAL loss = %.3f prec = %.3f' % (tot_loss/n, prec)
        logger.write(message+'\n')
        print message

if __name__ == "__main__":
    max_len = 10
    num_hid = 50
    batch_size = 16

    dp = DataPreprocessor()
    data = dp.preprocess('../data/tiny.train', '../data/tiny.test')
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), 
           len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))

    entityLSTM(mb_train, mb_test, len(data.chardict), max_len, num_hid)
    #entityMLP(mb_train, mb_test, len(data.chardict), max_len, num_hid)
    # generate random training data labeled with dot product with random weights,
    # weights and features scaled to +1 -1
    # def generateData(numExamples,numDims):
    #     def scaleToPlusOneMinusOne(m):
    #         return (m - 0.5) * 2.0
    #     x = scaleToPlusOneMinusOne( np.random.rand(numExamples,numDims) )
    #     targetWeights = scaleToPlusOneMinusOne( np.random.rand(numDims,1) )
    #     px = np.dot(x,targetWeights)
    #     return x,targetWeights,px
    
    # x,targetWeights,px = generateData(10000,2)

    # # now produce data for logistic regression
    # def sigma(x): return np.reciprocal( np.exp(-x) + 1.0 )
    # y0 = sigma(px)
    # y1 = ss.threshold(y0, 0.5, 1.0, 0.0)  # < 0.5 => 0
    # y2 = ss.threshold(y1, 0.0, 0.5, 1.0)   # > 0.5 => 1
    # y = np.hstack([y2, 1-y2])
    
    
    #LRCode(x, y, nullWeights)
    #bhuwan's code
    #linearRegressionCode(x, targetWeights, px)
    # bhuwanMLP(x,y)
    #bhuwanLSTM(x,y)
