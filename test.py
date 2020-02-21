
from neural_network.neuralNetwork import neuralNetwork

if __name__ == '__main__':
    #输入层,隐藏层,输出层的节点数量
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    #学习率
    learning_rate = 0.5
    #创建neuralNetwork的实例
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    print(n)