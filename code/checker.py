import mlp
import lstm
from functions import *
from utils import *
import argparse
from autograd import Autograd
EPS = 1e-4

def _crossEnt(x,y):
    log_x = np.nan_to_num(np.log(x))
    return - np.multiply(y,log_x).sum(axis=1, keepdims=True)

def fwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.eval(network.my_xman.operationSequence(network.my_xman.loss), valueDict)

def bwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.bprop(network.my_xman.operationSequence(network.my_xman.loss), valueDict,loss=np.float_(1.0))
def load_params_from_file(filename):
    return np.load(filename)[()]

def save_params_to_file(d, filename):
    np.save(filename, d)

def grad_check(network):
    # function which takes a network object and checks gradients
    # based on default values of data and params
    dataParamDict = network.my_xman.inputDict()
    fd = fwd(network, dataParamDict)
    grads = bwd(network, fd)
    for rname in grads:
        if network.my_xman.isParam(rname):
            fd[rname].ravel()[0] += EPS
            fp = network.fwd(fd)
            a = fp['loss']
            fd[rname].ravel()[0] -= 2*EPS
            fm = network.fwd(fd)
            b = fm['loss']
            fd[rname].ravel()[0] += EPS
            auto = grads[rname].ravel()[0]
            num = (a-b)/(2*EPS)
            if not np.isclose(auto, num, atol=1e-3):
                raise ValueError("gradients not close for %s, Auto %.5f Num %.5f"
                        % (rname, auto, num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--dataset', dest='dataset', type=str, default='autolab')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
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
    mb_test = MinibatchLoader(data.test, 200000, max_len, 
           len(data.chardict), len(data.labeldict))

    # build
    print "building mlp..."
    Mlp = mlp.MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    print "checking gradients..."
    grad_check(Mlp)

    mlp.main(params)
    targets = []
    indices = []
    for (idxs,e,l) in mb_test:
        targets.append(l)
        indices.extend(idxs)
    # print indices, targets
    # print np.vstack(targets)[indices]
    np.save("output2", np.vstack(targets)[indices])
    if 0 in indices:
        print "YAYA"
    print _crossEnt(np.load(output_file+".npy"), np.vstack(targets)[indices]).mean()
    # print "building lstm..."
    # lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    # print "done"
    # # check
    # print "checking gradients..."
    # grad_check(lstm)

