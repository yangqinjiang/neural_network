#definition
class neuralNetwork:
    # init neuralNetwork
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 分别设置输入层 ,隐藏层, 输出层的节点个数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        pass

    # train the neuralNetwork
    def train(self):
        pass

    #query the neuralNetwork
    def query(self):
        pass

