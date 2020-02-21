
from neural_network.neuralNetwork import neuralNetwork
import numpy as np

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

    #初始权重矩阵
    print(np.random.rand(3,3)) # 3x3的numpy数组,数组中的每个值都是0~1的随机值
    print(np.random.rand(3,3) - 0.5) # 减去0.5
    print("n.query...")
    print(n.query([1.0, 0.5, -1.5]))