# a sample use of xman for learning tasks
# 

import numpy as np
from xman import *
import struct
from autograd import *

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




    

    
