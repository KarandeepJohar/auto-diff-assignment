# some useful functions
import numpy as np
import scipy.stats as ss
from xman import *


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
    log_x = np.nan_to_num(np.log(x))
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
    'crossEnt-softMax': [lambda delta,out,x,y: delta*(_softMax(x)*y.sum(axis=1)[:,None] - y),  lambda delta,out,x,y:-delta*np.log(_softMax(x))],  #second one is never used for much
    'crossEnt':         [lambda delta,out,x,y: delta*((1.-y)/(1.-x)- (y/x)), lambda delta,out,x,y:delta*np.nan_to_num(np.log(1.-x)-np.log(x))],
    'tanh':             [lambda delta,out,x : delta * (1.0 - np.square(out))],
    'relu':             [lambda delta,out,x : delta * ((x>0).astype(np.float64))],
    'sigmoid':          [lambda delta,out,x : delta * out * (1.-out)],
    'hadamard':         [lambda delta,out,x1,x2: delta * x2,    lambda delta,out,x1,x2: delta * x1],
    }
