# a sample use of xman for learning tasks
# 

import numpy as np
import scipy.stats as ss
from xman import *
import struct


TRACE_EVAL = True
TRACE_BP = True

# some useful functions

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
    return - np.multiply(y,log_x)

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'mul':      lambda x1,x2: np.dot(x1,x2),
    'mean':     lambda x:x.mean(),
    'square':   np.square,
    'softMax':  _softMax,
    'crossEnt': _crossEnt,
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x:np.maximum(0,x),
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
# derivative defined for.  TODO: add derivatives for crossEnt and
# softMax for flexibility....

def _derivDot1(delta,out,x1,x2):
    return np.dot(delta, x2.transpose())

def _derivDot2(delta,out,x1,x2):
    return np.dot(x1.transpose(), delta)

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: delta],
    'subtract':         [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: -delta],
    'mul':              [_derivDot1, _derivDot2],
    'mean':             [lambda delta,out,x : delta * 1.0/float(x.shape[0])*np.ones(x.shape)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'crossEnt-softMax': [lambda delta,out,x,y: delta*(_softMax(x) - y),  lambda delta,out,x,y:-delta*x*y],  #second one is never used for much
    'tanh':             [lambda delta,out,x : delta * (1.0 - np.square(out))],
    'relu':             [lambda delta,out,x : delta * ((x>0).astype(np.float64))],
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

def MLP():

    x = XMan()
    # evaluating the model
    x.input = f.input()
    x.W1 = f.param()
    x.b1 = f.param()
    x.o1 = f.tanh( f.mul(x.input,x.W1) + x.b1 )
    x.W2 = f.param()
    x.b2 = f.param()
    x.o2 = f.tanh( f.mul(x.o1,x.W2) + x.b2 )
    x.output = f.softMax(x.o2)
    # loss
    x.y = f.input()
    x.loss = f.mean(f.crossEnt(x.output, x.y))
    return x.setup()


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

if __name__ == "__main__":

    # generate random training data labeled with dot product with random weights,
    # weights and features scaled to +1 -1
    epochs = 10
    def generateData(numExamples,numDims):
        def scaleToPlusOneMinusOne(m):
            return (m - 0.5) * 2.0
        x = scaleToPlusOneMinusOne( np.random.rand(numExamples,numDims) )
        targetWeights = scaleToPlusOneMinusOne( np.random.rand(numDims,1) )
        px = np.dot(x,targetWeights)
        return x,targetWeights,px
    
    x,targetWeights,px = generateData(10000,784)

    # simple gradient optimizer
    def learn(claz,epochs=epochs, rate=1.0, **initDict):
        def dvals(d,keys):
            return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
        h = claz()
        ad = Autograd(h)
        dataParamDict = h.inputDict(**initDict)
        opseq = h.operationSequence(h.loss)
        for i in range(epochs):
            vd = ad.eval(opseq,dataParamDict)
            print 'epoch:',i+1,dvals(vd,'loss'), error(vd['y'], vd['output']) 
            if vd['loss'] < 0.001:
                print 'good enough'
                break
            gd = ad.bprop(opseq,vd,loss=np.float_(1.0))
            for rname in gd:
                if h.isParam(rname):
                    if gd[rname].shape!=dataParamDict[rname].shape:
                        print rname, gd[rname].shape, dataParamDict[rname].shape
                    dataParamDict[rname] = dataParamDict[rname] - rate*gd[rname]
        return dataParamDict
    """h = LinearRegression()
    ad = Autograd(h)                
    nullWeights = np.zeros(targetWeights.shape)
    print 'clean training data, linear regression'
    fwd = learn(LinearRegression, epochs, rate=1.0, input=x, weights=nullWeights, y=px)
    fpd = ad.eval(h.operationSequence(h.output), h.inputDict(input=x[:10,:], weights=fwd['weights']))
    print 'learned/target weights:'
    print np.hstack([fwd['weights'], targetWeights])
   
    noisyPx = px + 0.1 * np.random.rand(px.shape[0], px.shape[1])
    print 'noisy training data, linear regression'
    fd = learn(LinearRegression, epochs, rate=1.0, input=x, weights=nullWeights, y=noisyPx)
    print 'learned/target weights:'
    print np.hstack([fd['weights'], targetWeights])
    
    # now produce data for logistic regression
    def sigma(x): return np.reciprocal( np.exp(-x) + 1.0 )
    y0 = sigma(px)
    y1 = ss.threshold(y0, 0.5, 1.0, 0.0)  # < 0.5 => 0
    y2 = ss.threshold(y1, 0.0, 0.5, 1.0)   # > 0.5 => 1
    y = np.hstack([y2, 1-y2])
    
    h = LogisticRegression()
    ad = Autograd(h)

    fwd = learn(LogisticRegression, epochs=50, input=x, rate=0.0005, weights=np.hstack([nullWeights, nullWeights]), y=y)
    fpd = ad.eval(h.operationSequence(h.output), h.inputDict(input=x, weights=fwd['weights']))
    print 'learned/target predictions, logistic regression'
    print np.hstack([fpd['output'], y])

    #bhuwan's code

    h = MLP()
    ad = Autograd(h)
    W1 = np.random.rand(784,5)
    b1=np.random.rand(5)
    W2=np.random.rand(5,2)
    b2=np.random.rand(2)

    print np.vstack([W1, b1])
    print np.vstack([W2, b2])


    fwd = learn(MLP, epochs=100, input=x, rate=0.01, 
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            y=y)
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(input=x, W1=fwd['W1'], b1=fwd['b1'], W2=fwd['W2'], b2=fwd['b2']))
    print 'learned/target predictions, MLP'
    print np.hstack([fpd['output'], y])
    print 'learned weights, biases'
    print np.vstack([fpd['W1'], fpd['b1']])
    print np.vstack([fpd['W2'], fpd['b2']])"""



    print "kkkkkkkkkkkkkkkkkk"
    # karan
    
    h = MLP2()
    ad = Autograd(h)
    X_train = parse_images("train-images.idx3-ubyte")
    y_train = parse_labels("train-labels.idx1-ubyte")
    X_test = parse_images("t10k-images.idx3-ubyte")#[:1000,:]
    y_test = parse_labels("t10k-labels.idx1-ubyte")#[:1000,:]
    # print X_test.shape


    # X_test = np.hstack([np.sum(y_test[:, :394], axis = 1, keepdims=True), np.sum(y_test[:, 394:], axis = 1, keepdims=True)])
    # y_test = np.hstack([np.sum(y_test[:, :5], axis = 1, keepdims=True), np.sum(y_test[:, 5:], axis = 1, keepdims=True)])
    # print "y_test", y_test.shape, y_test
    # print "x_test", X_test.shape
    # print np.sum(X_test, axis=1), X_test
    #nullWeights = np.zeros((X_test.shape[1], y_test.shape[1]))
    #fwd = learn(LogisticRegression, epochs=50, input=X_test, rate=0.000005, weights=nullWeights, y=y_test)
    # fwd = learn(MLP, epochs=10, input=X_test, rate=0.001, 
    #         W1=np.zeros((784,200)),
    #         b1=np.zeros((200)),
    #         W2=np.zeros((200,10)),
    #         b2=np.zeros((10)),
    #         y=y_test)
    # fwd = learn(MLP, epochs=10, input=X_test, rate=0.05, 
    #        W1=np.random.rand(784,100),
    #        b1=np.random.rand(100),
    #        W2=np.random.rand(100,2),
    #        b2=np.random.rand(2),
    #        # W3=np.random.rand(100,10),
    #        # b3=np.random.rand(10),
    #        y=y_test)
    W1 =0.1*np.random.rand(784,200)
    b1=0.1*np.random.rand(200)
    W2=0.1*np.random.rand(200,100)
    b2=0.1*np.random.rand(100)
    W3=0.1*np.random.rand(100,10)
    b3=0.1*np.random.rand(10)
    # W1 =np.zeros((784,200))
    # b1=np.zeros((200))
    # W2=np.zeros((200,100))
    # b2=np.zeros((100))
    # W3=np.zeros((100,10))
    # b3=np.zeros((10))
    # print np.vstack([W1, b1])
    # print np.vstack([W2, b2])
    fwd = learn(MLP2, epochs=1, input=X_train, rate=0.01, 
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            W3=W3,
            b3=b3,
            y=y_train)
    # print "kkkkkkkkkkkkkkkkkk"
    print X_test.shape, fwd['W1'].shape, fwd['b1'].shape, fwd['b1']
    print fwd['W2'].shape, fwd['b2'].shape
    print fwd['W3'].shape, fwd['b3'].shape 
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(input=X_test, W1=fwd['W1'], b1=fwd['b1'], W2=fwd['W2'], b2=fwd['b2'],W3=fwd['W3'], b3=fwd['b3']))
    # print 'learned/target predictions, MLP'
    # print np.hstack([fpd['output'], y_test])
    # print 'learned weights, biases'
    # print np.vstack([fwd['W1'], fwd['b1']])
    # print np.vstack([fwd['W2'], fwd['b2']])

    
