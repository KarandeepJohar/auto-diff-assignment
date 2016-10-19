import io
import numpy as np

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

