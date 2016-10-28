"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import time
import numpy as np
from xman import *
from utils import *
from autograd import *
from network import *

np.random.seed(0)
EPS = 1e-4

def glorot(m,n):
    # return scale for glorot initialization
    return np.sqrt(6./(m+n))

class MLP(Network):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
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
            self.params['W'+str(k)] = f.param(name='W'+str(k), 
                    default=sc*np.random.uniform(low=-1.,high=1.,
                        size=(layer_sizes[i], layer_sizes[i+1])))
            self.params['b'+str(k)] = f.param(name='b'+str(k), 
                    default=0.1*np.random.uniform(low=-1.,high=1.,size=(layer_sizes[i+1],)))
            
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

    def data_dict(self, X, y):
        dataDict = {}
        dataDict['X'] = X
        dataDict['y'] = y
        return dataDict

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('../data/%s.train'%dataset, '../data/%s.valid'%dataset, '../data/%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))

    # build
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    print "done"
    # check
    print "checking gradients..."
    # grad_check(mlp)
    print "ok"

    # train
    print "training..."
    logger = open('../logs/%s_mlp_L%d_H%d_B%d_E%d_lr%.3f.txt'%
            (dataset,max_len,num_hid,batch_size,epochs,init_lr),'w')
    tst = time.time()
    value_dict = mlp.graph.inputDict()
    max_prec = 0.
    for i in range(epochs):
        # learning rate schedule
        lr = init_lr/((i+1)**2)

        for (idxs,e,l) in mb_train:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = mlp.fwd(value_dict)
            gd = mlp.bwd(value_dict)
            value_dict = mlp.update(value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')

        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (idxs,e,l) in mb_valid:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd
            vd = mlp.fwd(value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += 1
        prec = evaluate(np.vstack(probs), np.vstack(targets))
        if prec>max_prec: max_prec = prec

        t_elap = time.time()-tst
        message = ('Epoch %d VAL loss %.3f prec %.3f max_prec %.3f time %.2f' % 
                (i,tot_loss/n,prec,max_prec,t_elap))
        logger.write(message+'\n')
        print message
    print "done"

    tot_loss, n= 0., 0
    probs = []
    targets = []
    indices = []
    for (idxs,e,l) in mb_test:
        # prepare input
        data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
        for k,v in data_dict.iteritems():
            value_dict[k] = v
        # fwd
        vd = mlp.fwd(value_dict)
        tot_loss += vd['loss']
        probs.append(vd['output'])
        targets.append(l)
        indices.extend(idxs)
        n += 1
    np.save(output_file, np.vstack(probs)[indices])
    print evaluate(np.vstack(probs), np.vstack(targets))
    print evaluate(np.load(output_file+".npy"), np.vstack(targets)[indices])

    prec = evaluate(np.vstack(probs), np.vstack(targets))
    if prec>max_prec: max_prec = prec

    t_elap = time.time()-tst
    message = ('Epoch %d VAL loss %.3f prec %.3f max_prec %.3f time %.2f' % 
            (i,tot_loss/n,prec,max_prec,t_elap))
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=10)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)