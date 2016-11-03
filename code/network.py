import numpy as np
from autograd import *
class Network:
    """
    Parent class with functions for doing forward and backward passes through the network, and
    applying updates. All networks should subclass this.
    """
    def display(self):
        print "Operation Sequence:"
        for o in self.op_seq:
            print o

    def fwd(self, valueDict):
        ad = Autograd(self.my_xman)
        return ad.eval(self.op_seq, valueDict)

    def bwd(self, valueDict):
        ad = Autograd(self.my_xman)
        return ad.bprop(self.op_seq, valueDict,loss=np.float_(1.0))

    def update(self, dataParamDict, grads, rate):
        for rname in grads:
            if self.my_xman.isParam(rname):
                if grads[rname].shape!=dataParamDict[rname].shape:
                    print rname, grads[rname].shape, dataParamDict[rname].shape
                dataParamDict[rname] = dataParamDict[rname] - rate*grads[rname]
        return dataParamDict
