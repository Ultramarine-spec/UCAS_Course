import random
import numpy as np
from tqdm import tqdm
from model import Net, MSE, SGD
import matplotlib.pyplot as plt

epochs = 100
sample_1 = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
                     [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
                     [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
                     [-0.76, 0.84, -1.96]])
sample_2 = np.array([[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
                     [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                     [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
                     [0.46, 1.49, 0.68]])
sample_3 = np.array([[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
                     [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
                     [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
                     [0.66, -0.45, 0.08]])

sample_1 = [(x, np.array([1, 0, 0])) for x in sample_1]
sample_2 = [(x, np.array([0, 1, 0])) for x in sample_2]
sample_3 = [(x, np.array([0, 0, 1])) for x in sample_3]

dataset = sample_1 + sample_2 + sample_3
random.shuffle(dataset)

batch_size = 1
X, Y = map(list, zip(*dataset))

layers = [
    {'type': 'linear', 'input_size': 3, 'output_size': 16},
    {'type': 'tanh'},
    {'type': 'linear', 'input_size': 16, 'output_size': 3},
    {'type': 'sigmoid'}
]
model = Net(layers)
criterion = MSE()
optimizer = SGD(model.parameters, lr=0.5)

loss_list = []
acc_list = []
for epoch in range(epochs):
    total_loss = 0
    total_right = 0
    for i in range(0, len(dataset), batch_size):
        end = min(i + batch_size, len(dataset))
        x = np.array(X[i:end])
        y = np.array(Y[i:end])

        output = model.forward(x)
        loss, right = criterion(output, y)
        delta = criterion.gradient()
        model.backward(delta)
        optimizer.update()

        total_loss += loss
        total_right += right
    loss_list.append(total_loss / len(dataset))
    acc_list.append(total_right / len(dataset))
    print(
        'epoch: {}, mean loss: {}, acc:{}, lr: {}'.format(epoch, total_loss / len(dataset), total_right / len(dataset),
                                                          optimizer.lr))
plt.figure()
plt.plot(range(len(loss_list)), loss_list, label="MSE loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(acc_list)), acc_list, label="Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
