# Neural network implementation
import pickle
import numpy as np
import gzip
import time
import random
import copy
import matplotlib.pyplot as plt
from datetime import datetime


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()


class NN(object):
    def __init__(self,
                 hidden_dims=(500, 400),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 data=None,
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.best_valid_accuracy = 0
        self.best_epoch = 0
        self.best_weights = None
        self.dims = (784, ) + hidden_dims + (10, )

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [],
                           'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
                                                )
        else:
            self.train, self.valid, self.test = data

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            if self.init_method == 'zero':
                self.weights[f"W{layer_n}"] = np.zeros(shape=(all_dims[layer_n - 1], all_dims[layer_n])).astype('float64')
            elif self.init_method == 'normal':
                self.weights[f"W{layer_n}"] = np.random.normal(0, 1, size=(all_dims[layer_n - 1], all_dims[layer_n]))
            elif self.init_method == "glorot":
                di = np.sqrt(6 / (all_dims[layer_n] + all_dims[layer_n - 1]))
                self.weights[f"W{layer_n}"] = np.random.uniform(low=-di,
                                                                high=di,
                                                                size=(all_dims[layer_n - 1], all_dims[layer_n])
                                                                )
            else:
                raise Exception("The provided initialization name is not valid.")

            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def show_parameters(self, title=None):
        if title:
            print(title)

        weights_count = 0
        bias_count = 0

        for layer in range(1, self.n_hidden + 2):
            print("Layer", layer, end=" - ")
            print("W", layer, self.weights[f"W{layer}"].shape, end="\t")
            print("b", layer, self.weights[f"b{layer}"].shape)

            weights_count += np.dot(np.shape(self.weights[f"W{layer}"])[0], np.shape(self.weights[f"W{layer}"])[1])
            bias_count += np.dot(np.shape(self.weights[f"b{layer}"])[0], np.shape(self.weights[f"b{layer}"])[1])

        print("Total number of parameters:", weights_count + bias_count, "\n")

    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            return x > 0
        # WRITE CODE HERE
        return np.maximum(0, x)

    def sigmoid(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            sig = self.sigmoid(x)
            return sig*(1 - sig)
        # WRITE CODE HERE
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            return 1 - (self.tanh(x)**2)
        # WRITE CODE HERE
        return np.tanh(x)

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        if x.ndim == 1:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        else:
            matrix = np.zeros((x.shape[0], x.shape[1]))
            for i, row in enumerate(x):
                matrix[i] = self.softmax(row)
        return matrix

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        for layer_n in range(1, self.n_hidden + 2):
            cache[f"A{layer_n}"] = cache[f"Z{layer_n - 1}"] @ self.weights[f"W{layer_n}"] + self.weights[f"b{layer_n}"]
            if layer_n == self.n_hidden + 1:
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
            else:
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        grads[f"dA{self.n_hidden + 1}"] = -(labels - output)

        for layer_n in reversed(range(1, self.n_hidden + 2)):
            grads[f"dW{layer_n}"] = cache[f"Z{layer_n - 1}"].T @ grads[f"dA{layer_n}"]
            grads[f"db{layer_n}"] = np.mean(grads[f"dA{layer_n}"], axis=0, keepdims=True)
            if layer_n > 1:
                grads[f"dZ{layer_n - 1}"] = grads[f"dA{layer_n}"] @ self.weights[f"W{layer_n}"].T
                grads[f"dA{layer_n - 1}"] = grads[f"dZ{layer_n - 1}"] * self.activation(cache[f"A{layer_n - 1}"], grad=True)

        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"W{layer}"] -= (self.lr / self.batch_size) * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= (self.lr / self.batch_size) * grads[f"db{layer}"]

    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        return -1 * np.sum(labels * np.log(prediction)) / prediction.shape[0]

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs, verbose=True):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            if valid_accuracy > self.best_valid_accuracy:
              self.best_valid_accuracy = valid_accuracy
              self.best_epoch = epoch
              self.best_weights = copy.deepcopy(self.weights)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

            if verbose:
                print(datetime.now(), "-", "Epoch", epoch + 1, ": loss =", self.train_logs["train_loss"][epoch])

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    def finite_difference(self, epsilon, layer, nb_parameters=10):
        x_valid, y_valid = self.valid
        self.batch_size = 1
        first_label = y_valid[0].reshape(1, -1)
        real_cache = self.forward(x_valid[0].reshape(1, -1))
        real_grad = self.backward(real_cache, first_label)
        gradient_approx = np.zeros(nb_parameters)
        gradient_real = np.zeros(nb_parameters)
        increment = 0
        for iline, line in enumerate(self.weights[f"W{layer}"]):
            for icol, col in enumerate(line):
                self.weights[f"W{layer}"][iline, icol] += epsilon
                plus_cache = self.forward(x_valid[0].reshape(1, -1))
                self.weights[f"W{layer}"][iline, icol] -= 2*epsilon
                minus_cache = self.forward(x_valid[0].reshape(1, -1))
                plus_loss = self.loss(plus_cache[f"Z{self.n_hidden + 1}"], first_label)
                minus_loss = self.loss(minus_cache[f"Z{self.n_hidden + 1}"], first_label)
                gradient_approx[increment] = (plus_loss - minus_loss) / (2*epsilon)
                gradient_real[increment] = real_grad[f"dW{layer}"][iline, icol]
                self.weights[f"W{layer}"][iline, icol] += epsilon
                increment += 1
                if increment == 10:
                  self.weights = copy.deepcopy(self.best_weights)
                  return gradient_approx, gradient_real, np.max(np.abs(gradient_approx - gradient_real))


""" Average loss measured on the training data at the end of each epoch """

initialization = ["zero", "normal", "glorot"]
n_epochs = 10
xrange = np.arange(n_epochs) + 1

for init in initialization:
    neural_network = NN(hidden_dims=(500, 400), seed=0, data=load_mnist(), init_method=init)
    neural_network.train_loop(n_epochs)
    neural_network.show_parameters(init + " initialization")

    plt.plot(xrange, neural_network.train_logs["train_loss"], label=(init + " initialization"))

plt.title("Average loss on the training data at the end of each epoch")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.xticks(xrange)
plt.show()
plt.savefig('initialization.png')


""" Grid search """

input_size = 784
learning_rates = [1e-1, 1e-2, 1e-3]
batch_sizes = [16, 32, 64]
hidden1 = [400, 500, 600]
hidden2 = [400, 500, 600]
activations = ["relu"]
output_size = 10
max_parameters = 1000000
min_parameters = 500000

combinations = list()
i = 1
for lr in learning_rates:
    for batch in batch_sizes:
        for h1 in hidden1:
            for h2 in hidden2:
                for activation in activations:
                    n_param = input_size*h1 + h1*h2 + output_size*h2 + h2 + input_size + output_size
                    if n_param < max_parameters:
                        if n_param > min_parameters:
                            combinations.append([lr, batch, h1, h2, activation])
                            print([lr, batch, h1, h2, activation])
                            i += 1

print(i-1)


for combination in combinations:
    learning_rate, batch, hidden, act = combination[0], combination[1], (combination[2], combination[3]), combination[4]

    neural_net = NN(lr=learning_rate, batch_size=batch, hidden_dims=hidden, seed=0, data=load_mnist(), activation=act)
    neural_net.train_loop(15, verbose=False)
    print("combination :", combination, "validation accuracy", neural_net.train_logs["validation_accuracy"][14])