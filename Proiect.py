import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pdb
import matplotlib.pyplot as plt

import pandas as pd



def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)


def plot_decision_boundary(X, y, W, b):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    # afisarea dreptei de decizie
    n_lines = W.shape[1]
    for i in range(n_lines):
        x1 = -0.5
        y1 = compute_y(x1, W[:, i], b[i])
        x2 = 0.5
        y2 = compute_y(x2, W[:, i], b[i])
        plt.plot([x1, x2], [y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


# derivata tanh
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2;


def backward_pass(a_1, a_2, z_1, W_2, X, Y, num_samples):
    # start calculating the slope of the loss function with the respect to z
    dz_2 = a_2 - y

    # then, with respect to our weights and biases
    dw_2 = (1 / num_samples) * np.matmul(a_1.T, dz_2)
    db_2 = (1 / num_samples) * np.sum(dz_2, axis=0)
    da_1 = np.matmul(dz_2, W_2.T)
    dz_1 = np.multiply(da_1, tanh_derivative(z_1))
    dw_1 = (1 / num_samples) * np.matmul(X.T, dz_1)
    db_1 = (1 / num_samples) * np.sum(dz_1, axis=0)

    return dw_1, db_1, dw_2, db_2


def forward_pass(X, W_1, b_1, W_2, b_2):
    # linear step
    z_1 = np.matmul(X, W_1) + b_1
    # activation tanh
    a_1 = np.tanh(z_1)

    # second linear step
    z_2 = np.matmul(a_1, W_2) + b_2
    # activation sigmoid
    a_2 = sigmoid(z_2)

    # return all values
    return z_1, a_1, z_2, a_2


def plot_decision(X_, W_1, W_2, b_1, b_2):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward_pass(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)

train_images = pd.read_csv("ml-unibuc-2019-24/train_samples.csv", dtype = 'double',header = None)
train_labels =pd.read_csv("ml-unibuc-2019-24/train_labels.csv", dtype = 'double',header = None)
test_images = pd.read_csv("ml-unibuc-2019-24/test_samples.csv", dtype='double',header = None)


#std_accuracies_test = get_accuracy_statistics(train_images, train_labels, test_images, 100, None)
#print("Training accuracies using STD normalization: ", std_accuracies_train)

#print("Test accuracies using STD normalization: ", std_accuracies_test )
#id=np.arange(1,5001)
#d={'Id': id, 'Prediction': std_accuracies_test}
#data_frame=pd.DataFrame(data=d)
#data_frame.to_csv('test_labels_none.csv')

# training set
#X = np.array([
    #[0, 0],
    #[0, 1],
    #[1, 0],
    #[1, 1]])
# labels
# or
#y = np.expand_dims(np.array([0, 1, 1, 0]), 1)

# one hidden layer with tanh as activation function
# one neuron in the output layer with sigmoid
num_hidden_neurons = 4096
num_output_neurons = 4096
W_1 = np.random.normal(0, 1, (4096, num_hidden_neurons))  # weights initilization
b_1 = np.zeros((num_hidden_neurons))
W_2 = np.random.normal(0, 1, (4096, num_hidden_neurons))  # weights initilization
b_2 = np.zeros((num_output_neurons))

num_samples = train_images.shape[0]

num_epochs = 250
lr = 0.5
for e in range(num_epochs):
    X, y = shuffle(train_images, train_labels)
    # forward
    z_1, a_1, z_2, a_2 = forward_pass(X, W_1, b_1, W_2, b_2)
    loss = -(y * np.log(a_2) + (1 - y) * np.log(1 - a_2)).mean()
    acc = (np.round(a_2) == y).mean()
    print('epoch: {} loss: {} accuracy: {}'.format(e, loss, acc))
    dw_1, db_1, dw_2, db_2 = backward_pass(a_1, a_2, z_1, W_2, X, y, num_samples)
    # plot_decision_boundary(X, np.squeeze(y), W_1, b_1) # use it when you have 2 neurons in the hidden layers
    plot_decision(X, W_1, W_2, b_1, b_2)
    W_1 -= lr * dw_1
    b_1 -= lr * db_1
    W_2 -= lr * dw_2
    b_2 -= lr * db_2



