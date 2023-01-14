import numpy as np
import matplotlib.pyplot as plt

w1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8],
               [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]])
w1 = np.insert(w1, 2, np.array([1] * 10).T, axis=1)
w2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9],
               [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]])
w2 = np.insert(w2, 2, np.array([1] * 10).T, axis=1)
w3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2],
               [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]])
w3 = np.insert(w3, 2, np.array([1] * 10).T, axis=1)
w4 = np.array([[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
               [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])
w4 = np.insert(w4, 2, np.array([1] * 10).T, axis=1)


def batch_perception(a, y, max_iter=10000, lr=1e-3, theta=1e-5):
    for i in range(max_iter):
        wrong_y = [yi for yi in y if np.dot(a, yi) <= 0]
        a = a + lr * sum(wrong_y)
        if np.sum(abs(lr * sum(wrong_y))) < theta:
            print("After {} steps. Converge! Number of misclassified samples: {}".format(i + 1, len(wrong_y)))
            print("Solution of a: {}\n".format(a))
            return a
    print("After {} steps. Not converge!".format(max_iter))
    print("Solution of a: {}\n".format(a))


def ho_kashyap(a, b, y, max_iter=10000, lr=1e-1, theta=1e-5):
    for i in range(max_iter):
        e = y @ a - b
        # print(np.sum(np.abs(e)))
        e_pos = 0.5 * (e + abs(e))
        b = b + 2 * lr * e_pos
        a = np.linalg.inv(y.T @ y) @ y.T @ b
        if np.sum(np.abs(e)) < theta:
            print("After {} steps. Converge! Error: {}".format(i + 1, np.sum(np.abs(e))))
            print("Solution of a: {},\nb: {}\n".format(a, b))
            return a, b
    print("After {} steps. Not converge! Error: {}".format(max_iter, np.sum(np.abs(e))))
    print("Solution of a: {},\nb: {}\n".format(a, b))
    return a, b


def MSE(w1, w2, w3, w4):
    X = np.c_[w1[:8].T, w2[:8].T, w3[:8].T, w4[:8].T]
    X_test = np.c_[w1[-2:].T, w2[-2:].T, w3[-2:].T, w4[-2:].T]
    Y = [[1, 0, 0, 0]] * 8 + [[0, 1, 0, 0]] * 8 + [[0, 0, 1, 0]] * 8 + [[0, 0, 0, 1]] * 8
    Y_test = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    Y = np.array(Y).T
    W = np.linalg.inv(X @ X.T) @ X @ Y.T

    pred = np.argmax(W.T @ X_test, axis=0) + 1
    accuracy = sum(pred == Y_test) / len(pred)
    print("W: {}".format(W))
    print("Accuracy: {}".format(accuracy))


def plot(dic, a):
    l1, l2 = list(dic.keys())
    p1, p2 = list(dic.values())
    x1 = p1[:, 0]
    y1 = p1[:, 1]

    x2 = p2[:, 0]
    y2 = p2[:, 1]
    plt.figure()
    plt.scatter(x1, y1, c='red', label=l1)
    plt.scatter(x2, y2, c='green', label=l2)
    if a[2] != 0:
        xd = np.arange(-10, 10, 0.01)
        yd = -a[0] * xd / a[1] - a[2] / a[1]
        plt.plot(xd, yd, linewidth='0.5')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


a0 = np.array([0, 0, 0])
b0 = np.array([1] * 20)
print("Batch Perception:")
y_w1w2 = np.concatenate((w1, -w2), axis=0)
y_w2w3 = np.concatenate((w2, -w3), axis=0)
a12 = batch_perception(a0, y_w1w2)
plot({"w1": w1, "w2": w2}, a12)
a23 = batch_perception(a0, y_w2w3)
plot({"w2": w2, "w3": w3}, a23)

print("Ho Kashyap:")
y_w1w3 = np.concatenate((w1, -w3), axis=0)
y_w2w4 = np.concatenate((w2, -w4), axis=0)
a13, _ = ho_kashyap(a0, b0, y_w1w3)
plot({"w1": w1, "w3": w3}, a13)
a24, _ = ho_kashyap(a0, b0, y_w2w4)
plot({"w2": w2, "w4": w4}, a24)

print("MSE:")
MSE(w1, w2, w3, w4)
