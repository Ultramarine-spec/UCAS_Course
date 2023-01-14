import numpy as np
import matplotlib.pyplot as plt

from k_means import k_means


def ng(X, k, sigma):
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = np.exp(-np.sum((X[i] - X[j]) ** 2) / (2 * sigma ** 2))
            W[j, i] = W[i, j]

    for i in range(n):
        idx = np.argsort(W[i])[::-1][k + 1:]
        W[i][idx] = 0

    W = (W + W.T) / 2

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    L_sym = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))

    eig_values, eig_vectors = np.linalg.eig(L_sym)

    idx = np.argsort(eig_values)[:2]
    eig_vectors = eig_vectors[:, idx]

    cluster, labels = k_means(eig_vectors, 2)

    return cluster, labels


with open('./data.txt', 'r') as f:
    data = [[eval(i) for i in x.strip().split(' ')] for x in f.readlines()]
    f.close()

X = np.array(data)

# plt.scatter(X[:100][:, 0], X[:100][:, 1])
# plt.scatter(X[100:][:, 0], X[100:][:, 1])
# plt.show()

acc = []
# for k in range(5, 50, 5):
#
#     cluster, label = ng(X, k, 1)
#
#     X1 = X[label == 0]
#     X2 = X[label == 1]
#
#     plt.figure()
#     plt.scatter(X1[:, 0], X1[:, 1], c='g')
#     plt.scatter(X2[:, 0], X2[:, 1], c='r')
#     plt.show()
#
#     correct = 0
#     for i in range(0, 200, 100):
#         correct += np.max(np.bincount(label[i:i + 100]))
#
#     print('Accuracy:', correct / len(label))
#     acc.append(correct / len(label))
#
# plt.figure()
# plt.plot(range(5, 50, 5), acc)
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.show()


for sigma in np.arange(0.1, 1, 0.1):

    cluster, label = ng(X, 50, sigma)

    X1 = X[label == 0]
    X2 = X[label == 1]

    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], c='g')
    plt.scatter(X2[:, 0], X2[:, 1], c='r')
    plt.show()

    correct = 0
    for i in range(0, 200, 100):
        correct += np.max(np.bincount(label[i:i + 100]))

    print('Accuracy:', correct / len(label))
    acc.append(correct / len(label))

plt.figure()
plt.plot(np.arange(0.1, 1, 0.1), acc)
plt.xlabel('sigma')
plt.ylabel('accuracy')
plt.show()
