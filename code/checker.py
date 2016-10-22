from char_lstm import *
def load_params_from_file(filename):
    return np.load(filename)[()]

def save_params_to_file(d, filename):
    np.save(filename, d)


if __name__ == '__main__':
    print "building mlp..."
    mlp = MLP()
    print "done"
    params = mlp.init_params(2,2)
    data = mlp.data_dict(np.random.rand(5,2),np.random.rand(5,2))
    _grad_check(mlp, data, params)
    dataParamDict = mlp.init_params(max_len*num_chars, train.num_labels)