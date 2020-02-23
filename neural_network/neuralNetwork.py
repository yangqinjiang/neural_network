import numpy
import scipy.special
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
        print("wih=")
        print(self.wih)
        print("who=")
        print(self.who)
        # 学习率
        self.lr = learningrate
        #sigmoid 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neuralNetwork
    def train(self,inputs_list,targets_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # calcualate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors ,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)) , numpy.transpose(hidden_outputs))
        # oupdate the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) ,numpy.transpose(inputs))
        pass

    #query the neuralNetwork
    # 只需要input_list,不需要任何其他输入
    def query(self,inputs_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calcualate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

