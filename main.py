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


class NeuralNetwork:
    def __init__(self, count_neurons):
        self.count_layers = len(count_neurons)
        self.count_neurons = count_neurons
        self.biases = [np.random.randn(y, 1) for y in count_neurons[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(
            count_neurons[:-1], count_neurons[1:])]

    def feedforward(self, a):
        a = np.array(a)[np.newaxis].T
        w = self.weights
        b = self.biases
        for i in range(len(w) - 1):
            a = relu(np.dot(w[i], a) + b[i])
        a = np.dot(w[-1], a) + b[-1]
        return a[0][0]

    def train(self, data_in, data_out, learning_rate, epochs):
        count_data = len(data_in)
        for epoch in range(epochs):
            w = self.weights
            b = self.biases
            a = []
            z = []
            nabla = []
            # 0 этап - вычисление средеквадратичной ошибки
            sum_error = 0
            for x, y in zip(data_in, data_out):
                sum_error += pow(y - self.feedforward(x), 2)
            sum_error = sum_error / (2 * count_data)
            print(f'Эпоха № {epoch}, ошибка = {sum_error}')
            # 1 этап - вычисление всех z и a - значений нейронов
            for x in data_in:
                a0 = np.array(x)[np.newaxis].T
                a_x = [a0]
                z_x = [a0]
                for i in range(len(w) - 1):
                    z0 = np.dot(w[i], a0) + b[i]
                    a0 = relu(z0)
                    z_x.append(z0)
                    a_x.append(a0)
                z0 = np.dot(w[-1], a0) + b[-1]
                a0 = z0
                z_x.append(z0)
                a_x.append(a0)
                a.append(a_x)
                z.append(z_x)
            # 2 этап - вычисление ошибки delta выходного слоя
            nabla_l = []
            dc = []
            for i in range(count_data):
                dc_da = a[i][-1][0] - data_out[i]
                dc.append([dc_da])
            dc = np.array(dc)
            da = []
            for z0 in z:
                if z0[-1][0][0] > 0:
                    da.append([[1]])
                else:
                    da.append([[0]])
            da = np.array(da)
            nabla_l = np.multiply(dc, da)
            nabla.append(nabla_l)
            # 3 этап - вычисление ошибки delta оставшихся слоев
            for l in range(self.count_layers - 2, 0, -1):
                delta_l1 = nabla_l
                nabla_l = []
                for x in delta_l1:
                    nabla_x = np.dot(w[l].T, x)
                    nabla_l.append(nabla_x)
                nabla_l = np.array(nabla_l)
                da = []
                for z0 in z:
                    z1 = []
                    for z00 in z0[l]:
                        if z00 > 0:
                            z1.append([1])
                        else:
                            z1.append([0])
                    da.append(z1)
                da = np.array(da)
                nabla_l = np.multiply(nabla_l, da)
                nabla.append(nabla_l)
            # 4 этап - изменение весов и смещений
            nabla.append(0)
            nabla.reverse()
            for l in range(self.count_layers - 1, 0, -1):
                for i in range(count_data):
                    w[l - 1] -= np.dot(nabla[l][i], a[i][l-1].T) * \
                        learning_rate / count_data
                    b[l - 1] -= nabla[l][i] * learning_rate / count_data


count_neurons = [3, 5, 1]
data_in, data_out = read_data('in.txt', count_neurons[0])
neural_network = NeuralNetwork(count_neurons)
neural_network.train(data_in, data_out, 0.1, 100)
prediction = neural_network.feedforward([0.707, 0.866, 0.966])
print(prediction)
