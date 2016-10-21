import io
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

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]


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

