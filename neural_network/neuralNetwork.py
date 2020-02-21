import numpy
# definition
class neuralNetwork:
    # init neuralNetwork
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 分别设置输入层 ,隐藏层, 输出层的节点个数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 链接权重矩阵 wih (weight of input_hidden) 和who (weight of hidden_output)
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # 正态概率的随机值, 第三个参数是numpy数组的形状大小
        self.wih = numpy.random.normal(0.0,pow(self.hnodes, -0.5), (self.hnodes , self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 学习率
        self.lr = learningrate

        pass

    # train the neuralNetwork
    def train(self):
        pass

    #query the neuralNetwork
    def query(self):
        pass

