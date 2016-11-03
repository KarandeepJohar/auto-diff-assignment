import mlp
import lstm
from functions import *
from utils import *
import argparse
from autograd import Autograd
import time
EPS = 1e-4
MLP_TIME_THRESHOLD = 15
LSTM_TIME_THRESHOLD = 100
ideal_mlp_loss = 0.9
ideal_lstm_loss = 0.9


def newIndices(indices):
    l = [0]*len(indices)
    for i in range(len(indices)):
        l[indices[i]]=i
    return l

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
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--mlp_init_lr', dest='mlp_init_lr', type=float, default=0.05)
    parser.add_argument('--lstm_init_lr', dest='lstm_init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--mlp-solution', dest='result_mlp_file', type=str, default='MLP_solution')
    parser.add_argument('--lstm-solution', dest='result_lstm_file', type=str, default='LSTM_solution')

    params = vars(parser.parse_args())
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    output_file = params['output_file']
    mlp_file = params['result_mlp_file']
    lstm_file = params['result_lstm_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('../data/%s.train'%dataset, '../data/%s.valid'%dataset, '../data/%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))

    result = {}
    result["mlp_grad_check"] = 0
    result["mlp_accuracy"] = 0
    result["mlp_time"] = 0
    result["lstm_grad_check"] = 0
    result["lstm_time"] = 0
    result["lstm_accuracy"] = 0

    try:
        # build
        print "building mlp..."
        Mlp = mlp.MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
        print "checking gradients..."
        grad_check(Mlp)
        result["mlp_grad_check"] = 15
    except Exception, e:
        print "GRADIENT CHECK FAILED"
        print e
        result["mlp_grad_check"] = 0


    t_start = time.time()
    params["init_lr"] = params["mlp_init_lr"]
    mlp.main(params)
    mlp_time = time.time()-t_start

    targets = []
    indices = []
    for (idxs,e,l) in mb_test:
        targets.append(l)
        indices.extend(idxs)
    student_mlp_loss = _crossEnt(np.load(output_file+".npy"), np.vstack(targets)[newIndices(indices)]).mean()
    # ideal_mlp_loss = _crossEnt(np.load(mlp_file+".npy"), np.vstack(targets)[newIndices(indices)]).mean()

    print "ideal_mlp_loss:", ideal_mlp_loss, "student_mlp_loss:", student_mlp_loss
    print ideal_mlp_loss/student_mlp_loss*10
    result["mlp_accuracy"] =  min(1,ideal_mlp_loss/student_mlp_loss)*10
        
    result["mlp_time"] = 15/max(mlp_time,15)*15
    
    try:
        # build
        print "building lstm..."
        Lstm = lstm.LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
        print "checking gradients..."
        grad_check(Lstm)
        result["lstm_grad_check"] = 15
    except Exception, e:
        print "GRADIENT CHECK FAILED"
        print e
        result["lstm_grad_check"] = 0

    t_start = time.time()
    params["init_lr"] = params["lstm_init_lr"]
    lstm.main(params)
    lstm_time = time.time()-t_start

    result["lstm_time"] = LSTM_TIME_THRESHOLD/max(lstm_time,LSTM_TIME_THRESHOLD)*15
    student_lstm_loss = _crossEnt(np.load(output_file+".npy"), np.vstack(targets)[newIndices(indices)]).mean()

    print "ideal_lstm_loss:", ideal_lstm_loss, "student_lstm_loss:", student_lstm_loss
    result["lstm_accuracy"] =   min(1,ideal_lstm_loss/student_lstm_loss)*10

    print result
    print "Your Autograder's total:", sum(result.values()), "/ 70\n";