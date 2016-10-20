# a sample use of xman for learning tasks
# 

import numpy as np
import scipy.stats as ss
from xman import *
import struct


TRACE_EVAL = False
TRACE_BP = False

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
    EPSILON = 10e-3
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
    'relu':             [lambda delta,out,x : delta * ((x>=0).astype(np.float64))],
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
    x.o1 = f.relu( f.mul(x.input,x.W1) + x.b1 )
    x.W2 = f.param()
    x.b2 = f.param()
    x.o2 = f.relu( f.mul(x.o1,x.W2) + x.b2 )
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

def MNIST():
    h = MLP2()
    ad = Autograd(h)
    X_train = parse_images("train-images.idx3-ubyte")
    y_train = parse_labels("train-labels.idx1-ubyte")
    X_test = parse_images("t10k-images.idx3-ubyte")#[:1000,:]
    y_test = parse_labels("t10k-labels.idx1-ubyte")#[:1000,:]
    
    W1=np.random.rand(X_train.shape[1],200)
    b1=np.random.rand(200)
    W2=np.random.rand(200,100)
    b2=np.random.rand(100)
    W3=np.random.rand(100,y_train.shape[1])
    b3=np.random.rand(y_train.shape[1])

    dataDict = {'input': X_train, 'y': y_train}
    fwd = learn(MLP2, dataDict, epochs=1, rate=10., batch_size=1000000000,
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

def LRCode(x,y, nullWeights):
    h = LogisticRegression()
    ad = Autograd(h)
    dataDict = {
            'input': x,
            'y': y,
            }
    fwd = learn(LogisticRegression, dataDict, epochs=50, batch_size=10000000, rate=0.5, weights=np.hstack([nullWeights, nullWeights]))
    fpd = ad.eval(h.operationSequence(h.output), h.inputDict(input=x, weights=fwd['weights']))
    print 'learned/target predictions, logistic regression'
    print np.hstack([fpd['output'], y])

def linearRegressionCode(x, targetWeights, px):
    h = LinearRegression()
    epochs =10
    ad = Autograd(h)                
    nullWeights = np.zeros(targetWeights.shape)
    print 'clean training data, linear regression'
    dataDict = {
            'input': x,
            'y': px,
            }
    fwd = learn(LinearRegression, dataDict, epochs=10, rate=1, batch_size=1000, weights=nullWeights)
    fpd = ad.eval(h.operationSequence(h.output), h.inputDict(input=x[:10,:], weights=fwd['weights']))
    print 'learned/target weights:'
    print np.hstack([fwd['weights'], targetWeights])
   
    noisyPx = px + 0.1 * np.random.rand(px.shape[0], px.shape[1])
    print 'noisy training data, linear regression'
    fd = learn(LinearRegression, dataDict, epochs=10, rate=1, batch_size=1000, weights=nullWeights)
    print 'learned/target weights:'
    print np.hstack([fd['weights'], targetWeights])

def learn(claz, dataDict, epochs=10, rate=1.0, batch_size=100, **initDict):
    def dvals(d,keys):
        return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
    x,y = dataDict['input'], dataDict['y']
    h = claz()
    ad = Autograd(h)
    dataParamDict = h.inputDict(**initDict)
    opseq = h.operationSequence(h.loss)
    epsilon = np.float_(0.00001)
    for i in range(epochs):
        for j in xrange(0,x.shape[0],batch_size):
            x_c = x[j:j+batch_size,:]
            y_c = y[j:j+batch_size,:]
            dataParamDict['input'] = x_c
            dataParamDict['y'] = y_c
            vd = ad.eval(opseq,dataParamDict)
            gd = ad.bprop(opseq,vd,loss=np.float_(1.0))

            for rname in gd:
                if h.isParam(rname):
                    ngd = np.ones(vd[rname].shape)
                    for p in xrange(vd[rname].shape[0]):
                        if len(vd[rname].shape)>1:
                            for q in xrange(vd[rname].shape[1]):
                                # print i,j
                                val = vd[rname][p][q]
                                vd[rname][p][q] = val+epsilon
                                ngd[p][q]=ad.eval(opseq, vd)["loss"]
                                vd[rname][p][q] = val-epsilon
                                ngd[p][q]-=ad.eval(opseq, vd)["loss"]
                                vd[rname][p][q] = val
                        else:
                            val = vd[rname][p]
                            vd[rname][p] = val+epsilon
                            ngd[p]=ad.eval(opseq, vd)["loss"]
                            vd[rname][p] = val-epsilon
                            ngd[p]-=ad.eval(opseq, vd)["loss"]
                            vd[rname][p] = val

                    print rname
                    print "GD:", gd[rname]
                    print "numerical", (ngd)/(2*epsilon)

                    

                    # if gd[rname].shape!=dataParamDict[rname].shape:
                    #     print rname, gd[rname].shape, dataParamDict[rname].shape

                    dataParamDict[rname] = dataParamDict[rname] - rate*gd[rname]
        print 'epoch:',i+1,dvals(vd,'loss'), error(vd['y'], vd['output']) 
        if vd['loss'] < 0.001:
            print 'good enough'
            break
    return dataParamDict

def bhuwanMLP(x,y):
    h = MLP()
    for op in h.operationSequence(h.loss):
        print op
    ad = Autograd(h)
    W1 = np.random.rand(x.shape[1],5)
    b1=np.random.rand(5)
    W2=np.random.rand(5,y.shape[1])
    b2=np.random.rand(y.shape[1])

    print np.vstack([W1, b1])
    print np.vstack([W2, b2])

    dataDict = {
            'input': x,
            'y': y,
            }
    fwd = learn(MLP, dataDict, epochs=1, rate=1, batch_size=1000,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2)
    fpd = ad.eval(h.operationSequence(h.output), 
            h.inputDict(input=x, W1=fwd['W1'], b1=fwd['b1'], W2=fwd['W2'], b2=fwd['b2']))
    print 'learned/target predictions, MLP'
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
    # linearRegressionCode(x, targetWeights, px)
    bhuwanMLP(x,y)

    # MNIST()




    

    
