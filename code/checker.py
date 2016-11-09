#!/usr/bin/env python

import mlp
import lstm
from functions import *
from utils import *
import argparse
from autograd import Autograd
import time
import os
EPS = 1e-4

MLP_LOSS_THRESHOLD = [(0.94,10), (1.02,5), (1.10, 0)]
LSTM_LOSS_THRESHOLD = [(0.92,10), (0.95,5), (1, 0)]
MLP_TIME_THRESHOLD = [(12,10), (20,5), (30,0)]
LSTM_TIME_THRESHOLD = [(100,10), (120,5), (130,0)]
def linear(thresholds, x):
    return float(thresholds[1][1]-thresholds[0][1])*(x- thresholds[0][0])/(thresholds[1][0]-thresholds[0][0])+thresholds[0][1]
""" x1 and x2 are two tuples """
def linear_mark(thresholds, x):
    if x<=thresholds[0][0]:
        return thresholds[0][1]
    elif x<=thresholds[1][0]:
        return linear(thresholds[:2], x)
    elif x<=thresholds[2][0]:
        return linear(thresholds[1:3], x)
    else:
        return thresholds[2][1]

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
            fp = fwd(network, fd)
            a = fp['loss']
            fd[rname].ravel()[0] -= 2*EPS
            fm = fwd(network, fd)
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
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='../data/autolab')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--mlp_init_lr', dest='mlp_init_lr', type=float, default=0.5)
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
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test.solution'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    result = {}
    result["mlp_grad_check"] = 0
    result["mlp_accuracy"] = 0

    result["mlp_time"] = 0
    result["lstm_grad_check"] = 0
    result["lstm_time"] = 0
    result["lstm_accuracy"] = 0
    targets = []
    indices = []
    for (idxs,e,l) in mb_test:
        targets.append(l)
        indices.extend(idxs)
    try:
        try:
            # build
            print "building mlp..."
            Mlp = mlp.MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
            print "checking gradients..."
            grad_check(Mlp)
            result["mlp_grad_check"] = 15
        except Exception, e:
            print "MLP GRADIENT CHECK FAILED"
            print e
            result["mlp_grad_check"] = 0


        t_start = time.clock()
        # print os.times()
        t_start1 = os.times()[0]
        params["init_lr"] = params["mlp_init_lr"]
        params["output_file"] = output_file+"_mlp"

        mlp.main(params)
        mlp_time = time.clock()-t_start
        user_time = os.times()[0]-t_start1
        # print os.times()

        student_mlp_loss = _crossEnt(np.load(params["output_file"]+".npy"), np.vstack(targets)).mean()
        # ideal_mlp_loss = _crossEnt(np.load(mlp_file+".npy"), np.vstack(targets)).mean()

        print "student_mlp_loss:", student_mlp_loss
        print "mlp_time:", mlp_time, "user_time:", user_time

        # print ideal_mlp_loss/student_mlp_loss*10
        result["mlp_accuracy"] =  linear_mark(MLP_LOSS_THRESHOLD, student_mlp_loss)
            
        result["mlp_time"] = linear_mark(MLP_TIME_THRESHOLD, mlp_time)
    except Exception, e:
        print "MLP CHECKING FAILED"
        print e
    try:
        try:
            # build
            print "building lstm..."
            Lstm = lstm.LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
            print "checking gradients..."
            grad_check(Lstm)
            result["lstm_grad_check"] = 15
        except Exception, e:
            print "LSTM GRADIENT CHECK FAILED"
            print e

        t_start = time.clock()
        t_start1 = os.times()[0]
        params["init_lr"] = params["lstm_init_lr"]
        params["output_file"] = output_file+"_lstm"

        lstm.main(params)
        lstm_time = time.clock()-t_start
        user_time = os.times()[0] - t_start1
        student_lstm_loss = _crossEnt(np.load(params["output_file"]+".npy"), np.vstack(targets)).mean()

        print  "student_lstm_loss:", student_lstm_loss
        print "lstm_time:", lstm_time, "user_time:", user_time
        result["lstm_accuracy"] =  linear_mark(LSTM_LOSS_THRESHOLD, student_lstm_loss)
            
        result["lstm_time"] = linear_mark(LSTM_TIME_THRESHOLD, lstm_time)


        scores = {}
        scores['scores'] = result
    except Exception, e:
        print "LSTM CHECKING FAILED"
        print e

    print "---------------------------------------------------";
    print "Your Autograder's total:", sum(result.values()), "/ 70";
    print "---------------------------------------------------";

    print "{ scores: {mlpgradcheck:"+str(result["mlp_grad_check"])+\
        ",lstmgradcheck:"+str(result["lstm_grad_check"])+\
        ",mlpaccuracy:"+str(result["mlp_accuracy"])+\
        ",lstmaccuracy:"+str(result["lstm_accuracy"])+\
        ",mlptime:"+str(result["mlp_time"])+\
        ",lstmtime:"+str(result["lstm_time"])+"} }"