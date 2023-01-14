import numpy as np


class Parameter:
    def __init__(self, data, requires_grad, skip_decay=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.skip_decay = skip_decay


class Linear:
    def __init__(self, input_size, output_size, requires_grad=True, bias=True, **kwargs):
        self.x = None
        self.W = Parameter(np.empty((input_size, output_size)), requires_grad)
        self.b = Parameter(np.empty(output_size), requires_grad) if bias else None
        self.requires_grad = requires_grad

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.W.data = np.random.uniform(-init_range, init_range, self.W.data.shape)
        self.b.data = np.zeros(self.b.data.shape)

    def forward(self, x):
        self.x = x if self.requires_grad else None
        y = np.dot(x, self.W.data)
        y = y + self.b.data if self.b is not None else y
        return y

    def backward(self, delta):
        """
        :param delta: shape[batch_size, output_size]
        :return: shape[batch_size, input_size]
        """
        if self.requires_grad:
            batch_size = delta.shape[0]
            self.W.grad = np.dot(self.x.T, delta) / batch_size
            assert self.W.grad.shape == self.W.data.shape
            if self.b is not None:
                self.b.grad = np.sum(delta, axis=0)
                assert self.b.grad.shape == self.b.data.shape
        return np.dot(delta, self.W.data.T)


class Tanh:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        self.y = (e_pos - e_neg) / (e_pos + e_neg)
        return self.y

    def backward(self, delta):
        """
        :param delta: shape[batch_size, output_size]
        :return: shape[batch_size, output_size]
        """
        return delta * (1 - self.y ** 2)


class Sigmoid:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, delta):
        """
        :param delta: shape[batch_size, output_size]
        :return: shape[batch_size, output_size]
        """
        return delta * self.y * (1 - self.y)


class MSE:
    def __init__(self):
        self.grad = None

    def gradient(self):
        return self.grad

    def __call__(self, x, y, requires_acc=True):
        assert x.shape == y.shape
        self.grad = x - y
        loss = 0.5 * np.sum((x - y) ** 2) / x.shape[0]
        if requires_acc:
            acc = np.argmax(x, axis=1) == np.argmax(y, axis=1)
            return loss, acc.sum()
        return loss


class Net:
    def __init__(self, layer_configs):
        self.layers = []
        self.parameters = []
        for config in layer_configs:
            self.layers.append(self.getLayer(config))

    def getLayer(self, config):
        t = config['type']
        if t == 'linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.b is not None:
                self.parameters.append(layer.b)
        elif t == 'tanh':
            layer = Tanh()
        elif t == 'sigmoid':
            layer = Sigmoid()
        else:
            raise TypeError
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class SGD:
    def __init__(self, parameters, lr, step_size=10000, gamma=1, decay=0):
        self.cnt = 0
        self.step_size = step_size
        self.gamma = gamma
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.decay_rate = 1.0 - decay

    def update(self):
        self.cnt += 1
        if self.cnt % self.step_size == 0:
            self.lr *= self.gamma
        for p in self.parameters:
            if self.decay_rate < 1 and not p.skip_decay:
                p.data *= self.decay_rate
            p.data -= self.lr * p.grad


class Adam:
    def __init__(self, parameters, lr, decay=0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.accumulated_beta1 = 1
        self.accumulated_beta2 = 1
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.eps = eps
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accumulated_grad_mom = [np.zeros(p.data.shape) for p in self.parameters]
        self.accumulated_grad_rms = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        self.accumulated_beta1 *= self.beta1
        self.accumulated_beta2 *= self.beta2
        lr = self.lr * ((1 - self.accumulated_beta2) ** 0.5) / (1 - self.accumulated_beta1)
        for p, grad_mom, grad_rms in zip(self.parameters, self.accumulated_grad_mom, self.accumulated_grad_rms):
            if self.decay_rate < 1 and not p.skip_decay: p.data *= self.decay_rate
            np.copyto(grad_mom, self.beta1 * grad_mom + (1 - self.beta1) * p.grad)
            np.copyto(grad_rms, self.beta2 * grad_rms + (1 - self.beta2) * np.power(p.grad, 2))
            p.data -= lr * grad_mom / (np.sqrt(grad_rms) + self.eps)
