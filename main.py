from typing import NewType
import numpy as np
from numpy.random.mtrand import randn


def read_data(filename, width):
    f = open(filename, 'r')
    line = f.read().strip()
    numbers = line.split(' ')
    data_in = np.zeros((len(numbers) - width, width))
    for i in range(len(numbers) - width):
        if i + 2 < len(numbers):
            for j in range(width):
                data_in[i, j] = float(numbers[i + j])
    data_out = np.array([float(numbers[i])
                        for i in range(width, len(numbers))])
    return data_in, data_out


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


class NeuralNetwork:
    def __init__(self, count_neurons):
        self.count_layers = len(count_neurons)
        self.count_neurons = count_neurons
        self.biases = [np.random.randn(y, 1) for y in count_neurons[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(count_neurons[:-1], count_neurons[1:])]

    def feedforward(self, a):
        a = np.array(a)[np.newaxis].T
        w = self.weights
        b = self.biases
        for i in range(len(w) - 1):
            a = relu(np.dot(w[i], a) + b[i])    
        a = np.dot(w[len(w) - 1], a) + b[len(w) - 1]
        return a
            

    def train(self, data_in, data_out, learning_rate, epochs):
        pass


count_neurons = [3, 4, 3, 1]
data_in, data_out = read_data('in.txt', count_neurons[0])
neural_network = NeuralNetwork(count_neurons)
prediction = neural_network.feedforward(data_in[0])
print(prediction)
