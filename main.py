import numpy as np


def read_data(filename, width):
    f = open(filename, 'r')
    line = f.read().strip()
    numbers = line.split(' ')
    print(numbers)
    data_in = np.zeros((len(numbers) - width, width))
    for i in range(len(numbers) - width):
        if i + 2 < len(numbers):
            for j in range(width):
                data_in[i, j] = float(numbers[i + j])
    data_out = np.array([float(numbers[i]) for i in range(width, len(numbers))])
    return data_in , data_out


def relu(x):
    return max(0, x)


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return relu(total)






data_in, data_out = read_data('in.txt', 3)




    
