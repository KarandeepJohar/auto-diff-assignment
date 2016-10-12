# a sample use of xman for an optimization task
# 

from xman import *

TRACE_EVAL = True
TRACE_BP = True

# functions I'll use for this problem

class f(XManFunctions):
    @staticmethod 
    def half(a): 
        return XManFunctions.registerDefinedByOperator('half',a)
    @staticmethod 
    def square(a): 
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod 
    def alias(a): 
        """ This will just make a copy of a register that
        has a different name."""
        return XManFunctions.registerDefinedByOperator('alias',a)

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'mul':      lambda x1,x2: x1*x2,
    'half':     lambda x: 0.5*x,
    'square':   lambda x: x*x,
    'alias':    lambda x: x,
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]

BP_FUNS = {
    'add':      [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: delta],
    'subtract': [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: -delta],
    'mul':      [lambda delta,out,x1,x2: delta*x2, lambda delta,out,x1,x2: delta*x1],
    'half':     [lambda delta,out,x: delta*0.5],
    'square':   [lambda delta,out,x: delta*2*x],
    'alias':    [lambda delta,out,x: delta],
    }

class Autograd(object):

    def __init__(self,xman):
        self.xman = xman

    def eval(self,opseq,valueDict):
        """ Evaluate the function specified by outputReg, where valueDict is a
        dict holding the values of any inputs/parameters that are
        needed (indexed by register name).
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
        associated with outputReg, find df/dg.  Here valueDict is a
        dict holding the values of any inputs/parameters that are
        needed for the gradient (indexed by register name), as returned
        by eval.
        """
        for (dstName,funName,inputNames) in reversed(opseq):
            delta = deltaDict[dstName]
            if TRACE_BP: print 'bprop [',delta,']',dstName,'=',funName,inputNames
            # values will be extended to include the next-level delta
            # and the output, and these will be passed as arguments
            values = [delta] + map(lambda a:valueDict[a], [dstName]+list(inputNames))
            for i in range(len(inputNames)):
                if TRACE_BP: print ' -',dstName,'->',funName,'-> (...',inputNames[i],'...)'
                result = (BP_FUNS[funName][i])(*values)
                # increment a running sum of all the delta's that are
                # pushed back to the i-th parameter, initializing to
                # zero if needed.
                self._incrementBy(deltaDict, inputNames[i], result)
        return deltaDict

    def _incrementBy(self, dict, key, inc):
        if key not in dict: dict[key] = 0
        dict[key] = dict[key] + inc

#
# some simple test cases
#

class Rect(XMan):
    h = f.input()
    w = f.input()
    area = h*w

class Cube(XMan):
    d = f.input()
    area = f.square(d)*d

class Triangle(XMan):
    h = f.input()
    w = f.input()
    area = f.half(h*w)

class House1(XMan):
    # a rectangle 'wall' and a triangular 'roof'
    hWall = f.input()
    wWall = f.input()
    wallArea = hWall * wWall
    hRoof = f.half(hWall)
    wRoof = f.alias(wWall)
    roofArea = f.half(hRoof * wRoof)
    area = roofArea + wallArea

# instance-level expressions

def House2():
    # like House1 but uses some macros and is instance-level
    x = XMan()
    # define some macros
    def rectArea(h,w): return h*w
    def roofHeight(wallHeight): return f.half(wallHeight)
    def triangleArea(h,w): return f.half(h*w)
    # declare the parameters
    x.h = f.param()
    x.w = f.param()
    x.target = f.input()
    x.area = rectArea(x.h,x.w) + triangleArea(roofHeight(x.h), x.w)
    # we will optimize square loss of area relative to some target
    # area given as input
    x.loss = f.square(x.area - x.target)
    return x.setup()

def House3():
    # like House2 but includes a target height for the wall
    x = XMan()
    # define some macros
    def rectArea(h,w): return h*w
    def roofHeight(wallHeight): return f.half(wallHeight)
    def triangleArea(h,w): return f.half(h*w)
    # declare the parameters
    x.h = f.param()
    x.w = f.param()
    x.target = f.input()
    x.targetHeight = f.input(8.0)
    x.heightFactor = f.input(100.0)  # relative weight of difference to target height in loss
    x.area = rectArea(x.h,x.w) + triangleArea(roofHeight(x.h), x.w)
    # we will optimize square loss of area relative to some target
    # area given as input
    x.loss = f.square(x.area - x.target) + f.square(x.h - x.targetHeight) * x.heightFactor
    return x.setup()

def House4():
    # like House1 but uses some macros and is instance-level
    x = XMan()
    # define some macros
    def rectArea(h,w): return h*w
    def roofHeight(wallHeight): return f.half(wallHeight)
    def triangleArea(h,w): return f.half(h*w)
    # declare the parameters
    x.h = f.param()
    x.w = f.param()
    x.target = f.input()
    x.area = rectArea(x.h,x.w) + triangleArea(roofHeight(x.h), x.w)
    # we will optimize square loss of area relative to some target
    # area given as input
    x.loss = f.square(x.area - x.target)
    return x.setup()

if __name__ == "__main__":

    # evaluate the area for a shape, and compute the gradient of each input
    def tryArea(XManClass,**initDict):
        xm = XManClass().setup()
        ad = Autograd(xm)
        opseq = xm.operationSequence(xm.area)
        vd = ad.eval(opseq,initDict)
        gd = ad.bprop(opseq,vd,area=1.0)
        def showDict(d):
            return "{"+",".join(sorted(map(lambda (k,v):"%s:%g" % (k,v), d.items())))+"}"
        print 'tryArea:',xm.__class__.__name__,'vd',showDict(vd),'gd',showDict(gd)

    # tryArea(Rect,h=10,w=5)
    # tryArea(Rect,h=2,w=10)
    # tryArea(Cube,d=3)
    # tryArea(Triangle,h=3,w=6)
    # tryArea(House1,hWall=6,wWall=20)
    #tryArea(House2,h=5,w=10)

    # simple gradient optimizer
    def fitArea(claz,epochs=10,**initDict):
        rate = 0.001
        def dvals(d,keys): return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
        h = claz()
        ad = Autograd(h)
        initDict = h.inputDict(**initDict)
        opseq = h.operationSequence(h.loss)
        for i in range(epochs):
            vd = ad.eval(opseq,initDict)
            print 'epoch',i+1,dvals(vd,'h w area target loss')
            if vd['loss'] < 0.01:
                print 'good enough'
                break
            gd = ad.bprop(opseq,vd,loss=1.0)
            for rname in gd:
                if h.isParam(rname):
                    initDict[rname] = initDict[rname] - rate*gd[rname]
                    
    fitArea(House2,10,h=5,w=10,target=200)
    # fitArea(House3,100,h=8,w=10,target=200)
