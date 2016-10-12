import unittest
import sys

from xman import *
class f(XManFunctions): 
    @staticmethod
    def tanh(a):
        return XManFunctions.registerDefinedByOperator('tanh',a)
    @staticmethod
    def logistic(a):
        return XManFunctions.registerDefinedByOperator('logistic',a)


class TestXMan(unittest.TestCase):

    def setUp(self):
        # test instance version
        net = XMan()
        net.input = f.input()
        net.input2hidden = f.param()
        net.bias1 = f.param()
        net.hidden2output = f.param()
        net.bias2 = f.param()
        net.hidden = f.tanh(net.input * net.input2hidden + net.bias1)
        net.output = f.tanh(net.hidden * net.hidden2output + net.bias2)
        self.net = net.setup()

    def testNamedRegs(self):
        net = self.net
        named = net.namedRegisterItems()
        regNames = map(lambda (rname,reg):rname, named)
        expected = "bias1 bias2 hidden hidden2output input input2hidden output".split()
        self.assertEqual(regNames, expected)
        
    def testRegs(self):
        net = self.net
        regNames = sorted(net.registers().keys())
        expected = "bias1 bias2 hidden hidden2output input input2hidden output z1 z2 z3 z4".split()
        self.assertEqual(regNames, expected)

    def testOpSeq1(self):
        opseq = self.net.operationSequence(self.net.output)
        print 'opsequence for MLP:'
        for k,op in enumerate(opseq):
            print k+1,op
        self.assertTrue(len(opseq)==6)
        outputRegseq = map(lambda op:op[0], opseq)
        expected = "z2 z1 hidden z4 z3 output".split()
        self.assertEqual(outputRegseq, expected)

if __name__=="__main__":
    if len(sys.argv)==1:
        unittest.main()
