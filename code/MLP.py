from utils import *
import time
import argparse
import io

class MLP:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate, funcs):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars*max_len
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.layer_sizes = [self.in_size, 100, 20, self.num_labels]
        self.funcs = funcs

        print "INITAILIZE:", self.in_size, self.num_labels

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
        self.params['W3'] = f.param()
        self.params['W3'].name = 'W3'
        self.params['b3'] = f.param()
        self.params['b3'].name = 'b3'

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
        x.o2 = f.tanh( f.mul(x.o1,self.params['W2']) + self.params['b2'] )
        x.o3 = f.tanh(f.mul(x.o2,self.params['W3']) + self.params['b3'])
        x.output = f.softMax(x.o3)
        # loss
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        self.graph = x.setup()
        print self.graph._registers
        self.display()

    def init_params(self):
        scale = 0.1
        W = [scale*np.random.randn(n,m) for m,n in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]
        b = [scale*np.random.randn(n) for n in self.layer_sizes[1:]]

        params = dict({"W"+str(k+1):W[k] for k in range(len(W))}, **{"b"+str(k+1):b[k] for k in range(len(b))}) 
        
        return params

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict



    # def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate, layer_sizes, funcs):
    #     self.num_chars = num_chars
    #     self.max_len = max_len
    #     self.in_size = num_chars*max_len
    #     self.batch_size = batch_size
    #     self.num_labels = num_labels
    #     self.layer_sizes = layer_sizes
    #     self.funcs = funcs

    #     if len(layer_sizes)-1!=len(funcs):
    #         raise ValueError("layer_sizes and funcs have incompataible sizes")
    #     # build network
    #     # ent_var = T.wmatrix('ent')
    #     # lab_var = T.imatrix('lab')
    #     # self.inps = [ent_var, lab_var]
    #     # probs, network = self.build_network()
    #     # params = L.get_all_params(network, trainable=True)
    #     def Layer2Hidden():

    #         x = XMan()
    #         # evaluating the model
    #         x.input = f.input()

    #         # for i in range(len(funcs)):

    #         x.W0 = f.param()
    #         x.b0 = f.param()
    #         x.o0 = f.tanh( f.mul(x.input,x.W0) + x.b0 )
    #         x.W1 = f.param()
    #         x.b1 = f.param()
    #         x.o1 = f.tanh( f.mul(x.o0,x.W1) + x.b1 )
    #         x.W2 = f.param()
    #         x.b2 = f.param()
    #         x.o2 = f.tanh( f.mul(x.o1,x.W2) + x.b2 )
    #         x.output = f.softMax(x.o2)
    #         # loss
    #         x.y = f.input()
    #         x.loss = f.mean(f.crossEnt(x.output, x.y))
    #         return x.setup()

    #     h = Layer2Hidden()
    #     for op in h.operationSequence(h.loss):
    #         print op
    #     ad = Autograd(h)

    #     layer_sizes = [self.in_size, 100, 20, self.num_labels]

    #     scale = 0.1
    #     W = [scale*np.random.randn(n,m) for m,n in zip(layer_sizes[:-1], layer_sizes[1:])]
    #     b = [scale*np.random.randn(n) for n in layer_sizes[1:]]

    #     params = dict({W+str(k):W[k] for k in range(len(W))}, **{b+str(k):b[k] for k in range(len(b))}) 

    #     # W1 = np.random.rand(self.in_size,100)
    #     # b1 = np.random.rand(100)
    #     # W2 = np.random.rand(100,20)
    #     # b2 = np.random.rand(20)
    #     # W3 = np.random.rand(20, self.num_labels)
    #     # b3 = np.random.rand(self.num_labels)
        
    #     # print np.vstack([W1, b1])
    #     # print np.vstack([W2, b2])

    #     dataDict = {
    #             'input': x,
    #             'y': y,
    #             }
    #     fwd = learn(MLP, dataDict, epochs=1, rate=1, batch_size=1000, **params)
    #     fpd = ad.eval(h.operationSequence(h.output), 
    #             h.inputDict(input=x, **{k:fpd[k] for k in params}))
    #     # print 'learned/target predictions, MLP'
    #     # print np.hstack([fpd['output'], y])
    #     # print 'learned weights, biases'
    #     # print np.vstack([fpd['W1'], fpd['b1']])
    #     # print np.vstack([fpd['W2'], fpd['b2']])
    #     # print fpd['W1'].shape, fpd['b1'].shape
    #     # print fpd['W2'].shape, fpd['b2'].shape



        # loss
        # loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        # updates = lasagne.updates.sgd(loss, params, learning_rate)
        # acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        # self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        # self.validate_fn = theano.function(self.inps, [loss,acc,probs])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1]*self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii*self.num_chars+idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def learn(self, dataDict, initDict, epochs=10, rate=1.0):
        def dvals(d,keys):
            return " ".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
        x,y = dataDict['X'], dataDict['y']
        h = self.graph
        ad = Autograd(h)
        dataParamDict = h.inputDict(**initDict)
        opseq = h.operationSequence(h.loss)
        dataParamDict['X'] = x
        dataParamDict['y'] = y
        
        for i in range(epochs):

            vd = ad.eval(opseq,dataParamDict)
            gd = ad.bprop(opseq,vd,loss=np.float_(1.0))
            for rname in gd:
                if h.isParam(rname):
                    if gd[rname].shape!=dataParamDict[rname].shape:
                        print rname, gd[rname].shape, dataParamDict[rname].shape
                    dataParamDict[rname] = dataParamDict[rname] - rate*gd[rname]
        print dvals(vd,'loss'), error(vd['y'], vd['output']) 
        # if vd['loss'] < 0.001:
        #     print 'good enough'
        #     break
        return dataParamDict

    # def validate(self, e, l):
    #     return self.validate_fn(self.to1hot(e), l)

    # def build_network(self):
    #     l_in = L.InputLayer(shape=(self.batch_size, self.in_size), input_var=self.inps[0])
    #     l_1 = L.DenseLayer(l_in, 100)
    #     l_2 = L.DenseLayer(l_1, 20)
    #     l_out = L.DenseLayer(l_2, self.num_labels, nonlinearity=lasagne.nonlinearities.softmax)
    #     p = L.get_output(l_out)
    #     return p, l_out



class LSTM:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars
        self.batch_size = batch_size
        self.num_labels = num_labels

        # build network
        ent_var = T.wtensor3('ent')
        lab_var = T.imatrix('lab')
        self.inps = [ent_var, lab_var]
        probs, network = self.build_network()
        params = L.get_all_params(network, trainable=True)

        # loss
        loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        updates = lasagne.updates.sgd(loss, params, learning_rate)
        acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        self.validate_fn = theano.function(self.inps, [loss,acc,probs])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1],self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii,idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def validate(self, e, l):
        return self.validate_fn(self.to1hot(e), l)

    def build_network(self):
        l_in = L.InputLayer(shape=(self.batch_size, self.max_len, self.in_size), 
                input_var=self.inps[0])
        l_1 = L.LSTMLayer(l_in, 50, backwards=False,
                only_return_final=False) # B x N x 50
        l_2 = L.DenseLayer(l_1, 100) # flattens N x D --> ND
        l_out = L.DenseLayer(l_1, self.num_labels, 
                nonlinearity=lasagne.nonlinearities.softmax)
        p = L.get_output(l_out)
        return p, l_out

class BiLSTM:
    def __init__(self, num_chars, max_len, batch_size, num_labels, learning_rate):
        self.num_chars = num_chars
        self.max_len = max_len
        self.in_size = num_chars
        self.batch_size = batch_size
        self.num_labels = num_labels

        # build network
        ent_var = T.wtensor3('ent')
        lab_var = T.imatrix('lab')
        self.inps = [ent_var, lab_var]
        probs, network = self.build_network()
        params = L.get_all_params(network, trainable=True)

        # loss
        loss = (lasagne.objectives.categorical_crossentropy(probs, lab_var)).mean()
        updates = lasagne.updates.sgd(loss, params, learning_rate)
        acc = (lasagne.objectives.categorical_accuracy(probs, lab_var)).mean()

        # functions
        self.train_fn = theano.function(self.inps, [loss,acc], updates=updates)
        self.validate_fn = theano.function(self.inps, [loss,acc,probs])

    def to1hot(self, index):
        # convert list of indices to one-hot representation
        out = np.zeros((index.shape[0],index.shape[1],self.num_chars), dtype='int16')
        for jj,item in enumerate(index):
            for ii,idx in enumerate(item):
                if idx==0: continue
                out[jj,ii,idx] = 1
        return out

    def train(self, e, l):
        return self.train_fn(self.to1hot(e), l)

    def validate(self, e, l):
        return self.validate_fn(self.to1hot(e), l)

    def build_network(self):
        l_in = L.InputLayer(shape=(self.batch_size, self.max_len, self.in_size), 
                input_var=self.inps[0])
        l_1 = L.LSTMLayer(l_in, 50, backwards=False,
                only_return_final=True) # B x 100
        l_2 = L.LSTMLayer(l_in, 50, backwards=True,
                only_return_final=True) # B x 100
        l_h = L.ConcatLayer([l_1,l_2])
        l_out = L.DenseLayer(l_h, self.num_labels, nonlinearity=lasagne.nonlinearities.softmax)
        p = L.get_output(l_out)
        return p, l_out

def evaluate(probs, targets):
    # compute precision @1
    preds = np.argmax(probs, axis=1)
    return float((targets[np.arange(targets.shape[0]),preds]==1).sum())/ \
            targets.shape[0]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, 
            default='MLP', help='Model to use')
    parser.add_argument('--learning_rate', dest='lr', type=float, 
            default=0.01, help='Learning rate')
    params = vars(parser.parse_args())

    dp = DataPreprocessor()
    data = dp.preprocess('../data/tiny.train','../data/tiny.test')
    mb_train = MinibatchLoader(data.training, 10, 10, len(data.labeldict))
    mb_test = MinibatchLoader(data.test, 10, 10, len(data.labeldict))

    m = eval(params['model'])(len(data.chardict), 10, 10, len(data.labeldict), params['lr'], [])

    init_dict = m.init_params()

    # fwd = learn(m, dataDict, init_dict)
    # fwd['X'] = x
    # fpd = ad.eval(h.operationSequence(h.output), 
    #         h.inputDict(**fwd))


    # logger = open('../logs/%s_%.3f_log.txt' % (params['model'].lower(),params['lr']),'w')
    max_prec = 0.
    # tst = time.time()
    first = True
    for epoch in range(100):
        print 'epoch ', epoch
        for (e,l) in mb_train:
            dataDict = {
                    'X': m.to1hot(e),
                    'y': l,
            }
            # print "data shape:", dataDict['X'].shape, dataDict['y'].shape 
            if first:
                fwd = m.learn(dataDict, init_dict)
            else:
                fwd = m.learn(dataDict, {k:fwd[k] for k in init_dict})
            
            
    #     tot_loss, tot_acc = 0., 0.
    #     n = 0
    #     probs = []
    #     targets = []
    #     for (e,l) in mb_test:
    #         loss, _, pr = m.validate(e,l)
    #         probs.append(pr)
    #         targets.append(l)
    #         tot_loss += loss
    #         n += 1
    #     prec = evaluate(np.vstack(probs), np.vstack(targets))
    #     if prec>max_prec: max_prec = prec
    #     message = 'VAL loss = %.3f prec = %.3f max_prec = %.3f' % (tot_loss/n, prec, max_prec)
    #     logger.write(message + '\n')
    #     print message
    # logger.write('Time elapsed = %.2f\n' % (time.time()-tst))
    # logger.close()