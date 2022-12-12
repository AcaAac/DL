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
        print("W shape is - ", self.W.shape)

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
        # print("y_i is - ", y_i)
        # x_i = x_i.reshape(x_i.shape[0], 1)
        # # print("x_i shape is - ", x_i.shape)
        # # print("x_i is - ", x_i)
        # # x_i = x_i.transpose()
        # x_bias = np.ones((x_i.shape[0], 1))
        # # print("x_bias is - ", x_bias, "his shape is - ", x_bias.shape)
        # phi_i = np.hstack((x_bias, x_i))
        # # print("Feature matrix is - ", phi_i)
        # w_1 = np.dot(phi_i.transpose(), y_i)
        # # print("2nd half of calculation is - ", w_1)
        # w_2 = np.dot(phi_i.transpose(), phi_i)
        # # print("without inverse 1st half of calculation is - ", w_2)
        # w_2 = np.linalg.inv(w_2)
        # # print("inverting we have - ", w_2)
        # self.W[y_i] = np.dot(w_2, w_1)
        # # self.W[y_i] = np.dot(np.dot(np.linalg.inv(np.dot(x_i.transpose(), x_i)), x_i.transpose()), y_i) 
        # In this case we are using cross-entropy as an error function
        # we are also assuming we have a binary logistic regression
        # Label scores according to the model (num_labels x 1).
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # weights init as normal distributition
        self.W1 = np.random.normal(loc = 0.1, scale = 0.01)
        self.W2 = np.random.normal(loc = 0.1, scale = 0.01)
        # biases init as zeros
        self.b1 = np.zeros((hidden_size))
        self.b2 = np.zeros((hidden_size))
        # init of pre/post activation of hidden layers
        # self.pre_act_H = np.zeros(hidden_size)
        # self.pos_act_H = np.zeros(hidden_size)
        self.hidden_size = hidden_size
    
    def ReLu(self, X):
        return max(0.0, X)
    def ReLu_deriv(self, X):
        if X >= 0:
            return 1
        else:
            return 0
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        # for sample in range(X.shape[0]):
        #     for unit in range(self.hidden_size):
        #         self.pre_act_H[unit] = np.dot(self.W[:,unit], X[sample,:])
        #         self.pos_act_H[unit] = ReLu(self.pre_act_H[unit])
        #     self.pre_act_O = np.dot(self.pre_act_H, self.W)
        #     self.pos_act_O = ReLu(pre_act_O)

            # cross-entropy error function
            # the same as for question above?
        temp = self.ReLu(np.dot(X, self.W1) + self.b1)
        y_hat = np.dot(temp, self.W2) + self.b2 
        return y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            y_hat = self.predict(x_i)
            # cross entropy

        # raise NotImplementedError
        pass 

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

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
