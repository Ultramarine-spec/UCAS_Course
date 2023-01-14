import numpy as np
import matplotlib.pyplot as plt


def k_means(X, k, max_iter=100):
    # 随机初始化 k 个聚类中心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for i in range(max_iter):
        # 计算每个样本点到聚类中心的距离
        distances = np.array([np.sqrt(np.sum((x - centroids) ** 2, axis=1)) for x in X])
        # 根据距离选择最近的聚类中心
        labels = np.argmin(distances, axis=1)
        # 计算每个聚类的新的聚类中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 如果聚类中心没有变化，则停止迭代
        if np.allclose(centroids, new_centroids):
            print(i)
            break
        centroids = new_centroids

    return centroids, labels


if __name__ == '__main__':

    cov = [[1, 0],
           [0, 1]]

    mean1 = [1, -1]
    x1 = np.random.multivariate_normal(mean1, cov, 200)
    mean2 = [5.5, -4.5]
    x2 = np.random.multivariate_normal(mean2, cov, 200)
    mean3 = [1, 4]
    x3 = np.random.multivariate_normal(mean3, cov, 200)
    mean4 = [6, 4.5]
    x4 = np.random.multivariate_normal(mean4, cov, 200)
    mean5 = [9, 0]
    x5 = np.random.multivariate_normal(mean5, cov, 200)

    X = np.concatenate((x1, x2, x3, x4, x5), axis=0)

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], c='g')
    plt.scatter(x2[:, 0], x2[:, 1], c='r')
    plt.scatter(x3[:, 0], x3[:, 1], c='b')
    plt.scatter(x4[:, 0], x4[:, 1], c='y')
    plt.scatter(x5[:, 0], x5[:, 1], c='c')
    plt.show()

    acc = []
    for _ in range(5):

        centroids, labels = k_means(X, k=5)

        print("Centers:")
        print(centroids)

        error = 0
        for m in [mean1, mean2, mean3, mean4, mean5]:
            error += np.min(np.sum((centroids - m) ** 2, axis=1))
        print("Error:", error)

        correct = 0
        for i in range(0, 1000, 200):
            correct += np.max(np.bincount(labels[i:i + 200]))

        print('Accuracy:', correct / len(X))
        acc.append(correct / len(X))

        # data_1 = X[labels == 0]
        # data_2 = X[labels == 1]
        # data_3 = X[labels == 2]
        # data_4 = X[labels == 3]
        # data_5 = X[labels == 4]
        #
        # plt.figure()
        # plt.scatter(data_1[:, 0], data_1[:, 1], c='g')
        # plt.scatter(data_2[:, 0], data_2[:, 1], c='r')
        # plt.scatter(data_3[:, 0], data_3[:, 1], c='b')
        # plt.scatter(data_4[:, 0], data_4[:, 1], c='y')
        # plt.scatter(data_5[:, 0], data_5[:, 1], c='c')
        # plt.show()

    plt.figure()
    plt.plot(acc)
    plt.show()
