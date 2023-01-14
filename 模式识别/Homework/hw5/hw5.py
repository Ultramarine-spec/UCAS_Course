from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def PCA(x, n_components):
    x_mean = np.mean(x, axis=0)
    x = x - x_mean
    cov = np.cov(x, rowvar=False)
    eig_values, eig_vectors = np.linalg.eigh(cov)
    eig_index = np.argsort(eig_values)[::-1][:n_components]
    filtered_eig_vec = eig_vectors[eig_index]

    return np.dot(x, filtered_eig_vec.T)


def LDA(x, y, n_components):
    x_mean = np.mean(x, axis=0)
    s_t = np.cov(x - x_mean, rowvar=False)

    c = len(np.unique(y))
    s_w = np.zeros((x.shape[1], x.shape[1]))
    for i in range(1, c + 1):
        x_i = x[y == i]
        x_i_mean = np.mean(x_i, axis=0)
        s_w += np.cov(x_i - x_i_mean, rowvar=False)

    s_b = s_t - s_w

    eig_values, eig_vectors = np.linalg.eigh(np.linalg.inv(s_w) @ s_b)
    eig_index = np.argsort(eig_values)[::-1][:n_components]
    filtered_eig_vec = eig_vectors[eig_index]

    return np.dot(x, filtered_eig_vec.T)


def fit_and_predict_pca(data, n_components):
    x = data[:, :-1]
    y = data[:, -1]

    x = PCA(x, n_components)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    y_pred = []
    for i in tqdm(x_test):
        _x = x_train - i
        _x = np.sum(np.square(_x), axis=1)
        idx = np.argmin(_x)
        y_pred.append(y_train[idx])

    acc = np.sum(y_pred == y_test) / len(y_test)

    print('Accuracy: ', acc)
    return acc


def fit_and_predict_lda(data, n_components):
    x = data[:, :-1]
    y = data[:, -1]

    x = LDA(x, y, n_components)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    y_pred = []
    for i in tqdm(x_test):
        _x = x_train - i
        _x = np.sum(np.square(_x), axis=1)
        idx = np.argmin(_x)
        y_pred.append(y_train[idx])

    acc = np.sum(y_pred == y_test) / len(y_test)

    print('Accuracy: ', acc)
    return acc


if __name__ == '__main__':
    face_data = loadmat('./ORLData_25.mat')['ORLData'].T
    vehicle_data = loadmat('./vehicle.mat')['UCI_entropy_data'].item()[4].T

    acc_face_pac = []
    acc_face_lda = []
    for i in range(10, 200, 5):
        acc_face_pac.append(fit_and_predict_pca(face_data, i))
        acc_face_lda.append(fit_and_predict_lda(face_data, i))

    acc_vehicle_pac = []
    acc_vehicle_lda = []
    for i in range(1, 18):
        acc_vehicle_pac.append(fit_and_predict_pca(vehicle_data, i))
        acc_vehicle_lda.append(fit_and_predict_lda(vehicle_data, i))

    # plot
    plt.figure()
    plt.plot(range(10, 200, 5), acc_face_pac, label='PCA')
    plt.plot(range(10, 200, 5), acc_face_lda, label='LDA')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.title('Face')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(1, 18), acc_vehicle_pac, label='PCA')
    plt.plot(range(1, 18), acc_vehicle_lda, label='LDA')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.title('Vehicle')
    plt.legend()
    plt.show()


