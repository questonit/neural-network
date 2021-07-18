import numpy as np

f = open('in.txt', 'r')
line = f.read().strip()
numbers = line.split(' ')
print(numbers)

width = 3

data_in = np.zeros((len(numbers) - width, width))
for i in range(len(numbers) - width):
    if i + 2 < len(numbers):
        for j in range(width):
            data_in[i, j] = float(numbers[i + j])
    
print(data_in)

data_out = np.array([float(numbers[i]) for i in range(width, len(numbers))])
print(data_out)

    
