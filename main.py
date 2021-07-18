from typing import NewType
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

def linear(x):
    return x


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs, activation):
        total = np.dot(self.weights, inputs) + self.bias
        return activation(total)


count_input = 3

data_in, data_out = read_data('in.txt', count_input)
count_layers = 3

count_neurons = [4, 3, 1]

neural_network = []
for i in range(count_layers):
    layer = []
    if i > 0:
        count = count_neurons[i - 1]
    else:
        count = count_input
    for num in range(count_neurons[i]):
        weights = [np.random.normal() for k in range(count)]
        bias = np.random.normal()
        neuron1 = Neuron(weights, bias)
        layer.append(neuron1)
    neural_network.append(layer)





    
