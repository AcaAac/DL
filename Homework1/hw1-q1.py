#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))
        # print("W shape is - ", self.W.shape)

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        y_hat = self.predict(x_i)
        if (y_hat != y_i):
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x_i 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x_i 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x_i num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]

def ReLu(x):
    return np.maximum(x, 0)
def ReLu_deriv(x):
    relu_der = np.empty(x.shape)
    for i in range(x.shape[0]):
        if x[i] > 0:
            relu_der[i] = 1
        else:
            relu_der[i] = 0
    return relu_der

class MLP(object):
    # Q2.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # weights init as normal distributition
        units = [n_features, hidden_size, n_classes]
        W1 = np.random.normal(loc = 0.1, scale = 0.1, size = (units[1], units[0]))
        b1 = np.zeros(units[1])
        W2 = np.random.normal(loc = 0.1, scale = 0.1, size = (units[2], units[1]))
        b2 = np.zeros(units[2])
        self.n_classes = n_classes
        self.weights = [W1, W2]
        self.biases = [b1, b2]

    def compute_loss(self, output, y_i):
        # print("output is - ", output)
        probs = np.exp(output) / np.sum(np.exp(output))
        loss = np.dot(-y_i, np.log(probs))
        return loss   
    def forward(self, x_i, weights, biases):
        num_layers = len(weights)
        hiddens = []
        g = ReLu
        for i in range(num_layers):
            h = x_i if i == 0 else hiddens[i-1]
            z = np.dot(weights[i], h) + biases[i]
            if i < num_layers - 1:  # Assume the output layer has no activation.
                hiddens.append(g(z))
        output = z
        # For classification this is a vector of logits (label scores).
        # For regression this is a vector of predictions.
        return output, hiddens

    def backward(self, x_i, y_i, output, hiddens, weights):
        num_layers = len(weights)
        g = ReLu
        z = output
        # softmax transformation.
        probs = np.exp(output) / np.sum(np.exp(output))
        grad_z = probs - y_i  # Grad of loss wrt last z.
        grad_weights = []
        grad_biases = []
        # print("grad_z is - ", grad_z)
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x_i if i == 0 else hiddens[i - 1]
            grad_weights.append(np.dot(grad_z[:,None], h[:,None].T))
            grad_biases.append(grad_z)

            # Gradient of hidden layer below.
            grad_h = np.dot(weights[i].T, grad_z)

            # Gradient of hidden layer below before activation.
            assert(g == ReLu)
            grad_z = grad_h * ReLu_deriv(h)
            # grad_z = grad_h * (1 - h ** 2)   # Grad of loss wrt z3.

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def update_parameters(self, weights, biases, grad_weights, grad_biases, learning_rate):
        num_layers = len(weights)
        for i in range(num_layers):
            weights[i] -= learning_rate * grad_weights[i]
            biases[i] -= learning_rate * grad_biases[i]
   
    def predict_label(self, output):
        # The most probable label is also the label with the largest logit.
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_labels = []
        for x_i in X:
            output, temp = self.forward(x_i, self.weights, self.biases)
            output = output - max(output)
            y_hat = self.predict_label(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        y_hat = np.argmax(y_hat,axis=1)
        # print("y is - ", y, "and y_hat is - ", y_hat)
        # print("y type is - ", type(y), "and y_hat type is - ", type(y_hat))
        # print("y shape is - ", y.shape, "and y_hat shape is - ", y_hat.shape)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        total_loss = 0
        # print("X shape is - ", X.shape)
        # print("y shape is - ", y.shape)
        onehot = np.zeros((np.size(y, 0), self.n_classes))
        # print("onehot is - ", onehot)
        for i in range(np.size(y, 0)):
                onehot[i, y[i]] = 1
        y = onehot
        # print("y is - ", y)
        for x_i, y_i in zip(X, y):
            output, hiddens = self.forward(x_i, self.weights, self.biases)
            # y_max = np.argmax(y_i)
            output = output - max(output)
            # print("y_i is - ", y_i)
            # print("output before compute_loss is - ", output)
            loss = self.compute_loss(output, y_i)
            # print("output after compute_loss is - ", output)
            total_loss += loss
            grad_weights, grad_biases = self.backward(x_i, y_i, output, hiddens, self.weights)
            self.update_parameters(self.weights, self.biases, grad_weights, grad_biases, learning_rate = learning_rate)
        print("Total loss: %f" % total_loss)
        return loss

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)
    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model

    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (model.evaluate(test_X, test_y)))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
