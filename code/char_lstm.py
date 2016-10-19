# a sample use of xman for learning tasks
# 

import numpy as np
import scipy.stats as ss
from xman import *
import struct
from utils import *

TRACE_EVAL = False
TRACE_BP = False

# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod
    def mean(a):
        return XManFunctions.registerDefinedByOperator('mean',a)
    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax',a)
    @staticmethod
    def crossEnt(a,b):
        return XManFunctions.registerDefinedByOperator('crossEnt',a,b)
    @staticmethod
    def tanh(a):
        return XManFunctions.registerDefinedByOperator('tanh',a)
    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu',a)
    @staticmethod
    def sigmoid(a):
        return XManFunctions.registerDefinedByOperator('sigmoid',a)
    @staticmethod
    def hadamard(a,b):
        return XManFunctions.registerDefinedByOperator('hadamard',a,b)

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

def _softMax(x):
    maxes = np.amax(x, axis=1)
    # print "line number 35", x.shape, maxes.shape
    maxes = maxes.reshape(maxes.shape[0], 1)
    # print "line number 37", maxes.shape
    e_x = np.exp(x - maxes)
    sums = np.sum(e_x, axis=1)
    # print "line number 40",  e_x.shape, sums.shape
    sums = sums.reshape(sums.shape[0], 1)
    # print "line number 42", sums.shape
    dist = e_x / sums
    return dist

def _crossEnt(x,y):
    EPSILON = 10e-5
    log_x = np.log(x + EPSILON)
    return - np.multiply(y,log_x).sum(axis=1, keepdims=True)

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'mul':      lambda x1,x2: np.dot(x1,x2),
    'mean':     lambda x:x.mean(),
    'square':   np.square,
    'softMax':  _softMax,
    'crossEnt': _crossEnt,
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(0,x),
    'sigmoid': lambda x: np.reciprocal(1.+np.exp(-x)),
    'hadamard': lambda x1,x2: x1*x2,
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# note: I don't have derivatives for crossEnt and softMax, instead we
# will look for patterns of the form "z = crossEnt(softMax(x), y)" and
# replace them with "z = crossEnt-softMax(x,y)", which we DO have a
# derivative defined for.  

def _derivDot1(delta,out,x1,x2):
    return np.dot(delta, x2.transpose())

def _derivDot2(delta,out,x1,x2):
    return np.dot(x1.transpose(), delta)

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0)
    else: return delta

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'mul':              [_derivDot1, _derivDot2],
    'mean':             [lambda delta,out,x : delta * 1.0/float(x.shape[0])*np.ones(x.shape)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'crossEnt-softMax': [lambda delta,out,x,y: delta*(_softMax(x)*y.sum(axis=1)[:,None] - y),  lambda delta,out,x,y:-delta*x*y],  #second one is never used for much
    'tanh':             [lambda delta,out,x : delta * (1.0 - np.square(out))],
    'relu':             [lambda delta,out,x : delta * ((x>0).astype(np.float64))],
    'sigmoid':          [lambda delta,out,x : delta * out * (1.-out)],
    'hadamard':         [lambda delta,out,x1,x2: delta * x2,    lambda delta,out,x1,x2: delta * x1],
    }

class Autograd(object):

    def __init__(self,xman):
        self.xman = xman

    def eval(self,opseq,valueDict):
        """ Evaluate the function defined by the operation sequence, where
        valueDict is a dict holding the values of any
        inputs/parameters that are needed (indexed by register name).
        """
        for (dstName,funName,inputNames) in opseq:
            if TRACE_EVAL: print 'eval:',dstName,'=',funName,inputNames
            inputValues = map(lambda a:valueDict[a], inputNames)
            fun = EVAL_FUNS[funName] 
            result = fun(*inputValues)
            valueDict[dstName] = result
        return valueDict

    def bprop(self,opseq,valueDict,**deltaDict):
        """ For each intermediate register g used in computing the function f
        associated with the opseq, find df/dg.  Here valueDict is a
        dict holding the values of any inputs/parameters that are
        needed for the gradient (indexed by register name), as
        returned by eval.
        """
        for (dstName,funName,inputNames) in self.optimizeForBProp(opseq):
            delta = deltaDict[dstName]
            if TRACE_BP: print 'bprop [',delta,']',dstName,'=',funName,inputNames
            # values will be extended to include the next-level delta
            # and the output, and these will be passed as arguments
            values = [delta] + map(lambda a:valueDict[a], [dstName]+list(inputNames))
            for i in range(len(inputNames)):
                if TRACE_BP: print ' -',dstName,'->',funName,'-> (...',inputNames[i],'...)'
                result = (BP_FUNS[funName][i])(*values)
                # increment a running sum of all the delta's that are
                # pushed back to the i-th parameter, initializing the
                # zero if needed.
                self._incrementBy(deltaDict, inputNames[i], result)
        return deltaDict

    def _incrementBy(self, dict, key, inc):
        if key not in dict: dict[key] = inc
        else: dict[key] = dict[key] + inc

    def optimizeForBProp(self,opseq):
        """ Optimize an operation sequence for backprop.  Currently, reverse
        it and replace any occurence of "z=crossEnt(a,b), ...,
        a=softMax(c)" with with "z=crossEnt-softMax(c,b)"
        """
        opseq = list(reversed(opseq))
        # find where z = f(...) appears
        def find(dst=None,fun=None):
            def match(actual,target): return target==None or actual==target
            for k,(dstName,funName,inputNames) in enumerate(opseq):
                if match(dstName,dst) and match(funName,fun):
                    return k
            return -1
        # look for places to optimize
        crossEntOptimizations = []
        for k,(dstName,funName,inputNames) in enumerate(opseq):
            # look for z=crossEnt(softMax(p), y) where y is an input or param
            if funName=='crossEnt':
                (a,b) = inputNames; ka = find(dst=a); kb = find(dst=b)
                if ka>=0 and kb<0 and opseq[ka][1]=='softMax':
                    crossEntOptimizations.append((k,ka))
        # perform the optimization, by splicing out operation index ka
        # and replacing operation k with a single crossEnt-softMax
        # operation
        for (k,ka) in crossEntOptimizations:
            z = opseq[k][0]
            b = opseq[k][2][1]
            c = opseq[ka][2][0]
            opseq = opseq[:k] + [(z,'crossEnt-softMax',(c,b))] + opseq[k+1:ka]+opseq[ka+1:]
        return opseq
#
# some test cases
#

class LSTM:

    def __init__(self, L):
        self.length = L
        self._declareParams()
        self._declareInputs()
        self.build()

    def display(self):
        for o in self.graph.operationSequence(self.graph.loss):
            print o

    def _declareParams(self):
        self.params = {}
        self.params['Wi'] = f.param()
        self.params['Wi'].name = 'Wi'
        self.params['Ui'] = f.param()
        self.params['Ui'].name = 'Ui'
        self.params['bi'] = f.param()
        self.params['bi'].name = 'bi'
        self.params['Wf'] = f.param()
        self.params['Wf'].name = 'Wf'
        self.params['Uf'] = f.param()
        self.params['Uf'].name = 'Uf'
        self.params['bf'] = f.param()
        self.params['bf'].name = 'bf'
        self.params['Wo'] = f.param()
        self.params['Wo'].name = 'Wo'
        self.params['Uo'] = f.param()
        self.params['Uo'].name = 'Uo'
        self.params['bo'] = f.param()
        self.params['bo'].name = 'bo'
        self.params['Wc'] = f.param()
        self.params['Wc'].name = 'Wc'
        self.params['Uc'] = f.param()
        self.params['Uc'].name = 'Uc'
        self.params['bc'] = f.param()
        self.params['bc'].name = 'bc'
        self.params['W1'] = f.param()
        self.params['W1'].name = 'W1'
        self.params['b1'] = f.param()
        self.params['b1'].name = 'b1'

    def _declareInputs(self):
        self.inputs = {}
        for i in range(self.length):
            self.inputs['input_%d'%i] = f.input()
            self.inputs['input_%d'%i].name = 'input_%d'%i
        self.inputs['y'] = f.input()
        self.inputs['y'].name = 'y'

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
        x.hid_init = f.input()
        x.cell_init = f.input()
        hh = x.hid_init
        cc = x.cell_init
        for i in range(self.length):
            hh, cc = LSTM._step(self.inputs['input_%d'%i], hh, cc, self.params)
        x.o1 = f.relu( f.mul(hh,self.params['W1']) + self.params['b1'] )
        x.output = f.softMax(x.o1)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        self.graph = x.setup()
        self.display()

    def init_params(self, in_size, num_hidden, out_size):
        paramDict = {}
        paramDict['Wi'] = np.random.rand(in_size, num_hidden)
        paramDict['Ui'] = np.random.rand(num_hidden, num_hidden)
        paramDict['bi'] = np.random.rand(num_hidden)
        paramDict['Wo'] = np.random.rand(in_size, num_hidden)
        paramDict['Uo'] = np.random.rand(num_hidden, num_hidden)
        paramDict['bo'] = np.random.rand(num_hidden)
        paramDict['Wf'] = np.random.rand(in_size, num_hidden)
        paramDict['Uf'] = np.random.rand(num_hidden, num_hidden)
        paramDict['bf'] = np.random.rand(num_hidden)
        paramDict['Wc'] = np.random.rand(in_size, num_hidden)
        paramDict['Uc'] = np.random.rand(num_hidden, num_hidden)
        paramDict['bc'] = np.random.rand(num_hidden)
        paramDict['W1'] = np.random.rand(num_hidden, out_size)
        paramDict['b1'] = np.random.rand(out_size)
        return paramDict

class MLP:

    def __init__(self):
        self._declareParams()
        self._declareInputs()
        self._build()

    def display(self):
        for o in self.graph.operationSequence(self.graph.loss):
            print o

    def _declareParams(self):
        self.params = {}
        self.params['W1'] = f.param()
        self.params['W1'].name = 'W1'
        self.params['b1'] = f.param()
        self.params['b1'].name = 'b1'
        self.params['W2'] = f.param()
        self.params['W2'].name = 'W2'
        self.params['b2'] = f.param()
        self.params['b2'].name = 'b2'

    def _declareInputs(self):
        self.inputs = {}
        self.inputs['X'] = f.input()
        self.inputs['X'].name = 'X'
        self.inputs['y'] = f.input()
        self.inputs['y'].name = 'y'

    def _build(self):
        x = XMan()
        # evaluating the model
        x.o1 = f.tanh( f.mul(self.inputs['X'],self.params['W1']) + self.params['b1'] )
        x.o2 = f.relu( f.mul(x.o1,self.params['W2']) + self.params['b2'] )
        x.output = f.softMax(x.o2)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        self.graph = x.setup()
        print self.graph._registers
        self.display()

    def init_params(self, in_size, out_size):
        paramDict = {}
        paramDict['W1'] = np.random.rand(in_size,5)
        paramDict['b1'] = np.random.rand(5)
        paramDict['W2'] = np.random.rand(5,out_size)
        paramDict['b2'] = np.random.rand(out_size)
        return paramDict

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict

def MLP2():

    x = XMan()
    # evaluating the model
    x.input = f.input()
    x.W1 = f.param()
    x.b1 = f.param()
    x.o1 = f.relu( x.input*x.W1 + x.b1 )
    x.W2 = f.param()
    x.b2 = f.param()
    x.o2 = f.relu( x.o1*x.W2 + x.b2 )
    x.W3 = f.param()
    x.b3 = f.param()
    x.o3 = f.relu( x.o2*x.W3 + x.b3 )
    x.output = f.softMax(x.o3)
    # loss
    x.y = f.input()
    x.loss = f.mean(f.crossEnt(x.output, x.y))
    return x.setup()

def LinearRegression():

    x = XMan()
    # evaluating the model
    x.input = f.input()
    x.weights = f.param()
    x.output = x.input * x.weights
    # the loss
    x.y = f.input()
    x.loss = f.mean(f.square(x.output - x.y))
    return x.setup()

def LogisticRegression():

    x = XMan()
    # evaluating the model
    x.input = f.input()
    x.weights = f.param()
    x.output = f.softMax( x.input * x.weights )
    # the loss
    x.y = f.input()
    x.loss = f.mean(f.crossEnt(x.output, x.y))
    return x.setup()

def parse_images(filename):
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    sx,sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im =  struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/255.0 for x in im]);
    return np.array(X);

def parse_labels(filename):
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], 
                                    dtype=np.float64)
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    return one_hot(np.array(struct.unpack('B'*size, f.read(size))), 10)

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

def MNIST():
    h = MLP2()
    ad = Autograd(h)
    X_train = parse_images("train-images.idx3-ubyte")
    y_train = parse_labels("train-labels.idx1-ubyte")
    X_test = parse_images("t10k-images.idx3-ubyte")#[:1000,:]
    y_test = parse_labels("t10k-labels.idx1-ubyte")#[:1000,:]
    
    W1 =0.01*np.random.rand(X_train.shape[1],200)
    b1=0.01*np.random.rand(200)
    W2=0.01*np.random.rand(200,100)
    b2=0.01*np.random.rand(100)
    W3=0.01*np.random.rand(100,y_train.shape[1])
    b3=0.01*np.random.rand(y_train.shape[1])

    dataDict = {
            'input': X_train,
            'y': y_train,
            }
    fwd = learn(MLP2, dataDict, epochs=100, rate=1., batch_size=100,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            W3=W3,
            b3=b3)
    # print "kkkkkkkkkkkkkkkkkk"
    print X_test.shape, fwd['W1'].shape, fwd['b1'].shape, fwd['b1']
    print fwd['W2'].shape, fwd['b2'].shape
    print fwd['W3'].shape, fwd['b3'].shape 
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(input=X_test, W1=fwd['W1'], b1=fwd['b1'], W2=fwd['W2'], b2=fwd['b2'],W3=fwd['W3'], b3=fwd['b3']))

def learn(net, dataDict, initDict, epochs=10, rate=1.0, batch_size=100):
    def dvals(d,keys):
        return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
    x,y = dataDict['input'], dataDict['y']
    h = net.graph
    ad = Autograd(h)
    dataParamDict = h.inputDict(**initDict)
    opseq = h.operationSequence(h.loss)
    for i in range(epochs):
        for j in range(0,x.shape[0],batch_size):
            x_c = x[j:j+batch_size,:]
            y_c = y[j:j+batch_size,:]
            dataParamDict['input_0'] = x_c
            dataParamDict['input_1'] = x_c
            dataParamDict['input_2'] = x_c
            dataParamDict['y'] = y_c
            dataParamDict['hid_init'] = np.zeros((100,5))
            dataParamDict['cell_init'] = np.zeros((100,5))
            vd = ad.eval(opseq,dataParamDict)
            gd = ad.bprop(opseq,vd,loss=np.float_(1.0))
            for rname in gd:
                if h.isParam(rname):
                    if gd[rname].shape!=dataParamDict[rname].shape:
                        print rname, gd[rname].shape, dataParamDict[rname].shape
                    dataParamDict[rname] = dataParamDict[rname] - rate*gd[rname]
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
    fwd = learn(mlp, dataDict, init_dict, epochs=50, rate=1, batch_size=100)
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
    lstm = LSTM(2)
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

if __name__ == "__main__":

    # generate random training data labeled with dot product with random weights,
    # weights and features scaled to +1 -1
    def generateData(numExamples,numDims):
        def scaleToPlusOneMinusOne(m):
            return (m - 0.5) * 2.0
        x = scaleToPlusOneMinusOne( np.random.rand(numExamples,numDims) )
        targetWeights = scaleToPlusOneMinusOne( np.random.rand(numDims,1) )
        px = np.dot(x,targetWeights)
        return x,targetWeights,px
    
    x,targetWeights,px = generateData(10000,2)
    nullWeights = np.zeros(targetWeights.shape)


    
    
    # now produce data for logistic regression
    def sigma(x): return np.reciprocal( np.exp(-x) + 1.0 )
    y0 = sigma(px)
    y1 = ss.threshold(y0, 0.5, 1.0, 0.0)  # < 0.5 => 0
    y2 = ss.threshold(y1, 0.0, 0.5, 1.0)   # > 0.5 => 1
    y = np.hstack([y2, 1-y2])
    
    
    #LRCode(x, y, nullWeights)
    #bhuwan's code
    #linearRegressionCode(x, targetWeights, px)
    #bhuwanMLP(x,y)
    bhuwanLSTM(x,y)

    #MNIST()




    

    
