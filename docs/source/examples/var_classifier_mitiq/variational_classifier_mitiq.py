# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:08:07 2021

@author: misty

Example of variational problem made trainable wtih PyQuil and
noise mitigation toolkit Mitiq

Variational classifier problem setup based on
PennyLane variational classifier tutorial:
https://pennylane.ai/qml/demos/tutorial_variational_classifier.html,
and PennyLane quantum gradient tutorial:
https://pennylane.ai/qml/demos/tutorial_backprop.html,

"""

from pyquil import get_qc, Program
from pyquil.gates import RY, RZ, X, CNOT
from mitiq import zne, execute_with_zne
from mitiq.mitiq_pyquil.pyquil_utils import (
    ground_state_expectation,
    generate_qcs_executor,
)
import numpy as np
import matplotlib.pyplot as plt

# initialize quantum device
qpu = get_qc("2q-noisy-qvm")
shots = 1
# create function to calculate expectation value
executor_fn = generate_qcs_executor(qpu, ground_state_expectation, shots)

# initialize quanutm circuit
program = Program()


# Translate input x into rotation angles for state preparation
def calc_angles(x):

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2)
                          / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2)
                          / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# Prepare the quantum state with inputs encoded as rotation angles
def stateprep(program, a):
    program += RY(angle=a[0], qubit=0)
    program += CNOT(0, 1)
    program += RY(angle=a[1], qubit=1)
    program += CNOT(0, 1)
    program += RY(angle=a[2], qubit=1)
    program += X(qubit=0)
    program += CNOT(0, 1)
    program += RY(angle=a[3], qubit=1)
    program += CNOT(0, 1)
    program += RY(angle=a[4], qubit=1)
    program += X(qubit=0)
    return program


def layer(program, W):
    program += RZ(W[0, 0], qubit=0)
    program += RY(W[0, 1], qubit=0)
    program += RZ(W[0, 2], qubit=0)
    program += RZ(W[1, 0], qubit=1)
    program += RY(W[1, 1], qubit=1)
    program += RZ(W[1, 2], qubit=1)
    program += CNOT(0, 1)
    return program


def circuit(program, weights, angles, mitigate):
    program = stateprep(program, angles)

    for W in weights:
        program = layer(program, W)
    if mitigate is True:
        fac = zne.inference.LinearFactory(scale_factors=[1.0, 3.0])
        result = execute_with_zne(program, executor_fn, fac)
    else:
        result = executor_fn(program)
    return result


# Define loss function as a standard square loss function
def square_loss(labels, predictions):
    loss = 0
    for la, p in zip(labels, predictions):
        loss = loss + (la - p) ** 2
    loss = loss / len(labels)
    return loss


# Define the accuracy given target labels and model predictions, to monitor
# how many inputs the classifier predicted correctly
def accuracy(labels, predictions):

    loss = 0
    for la, p in zip(labels, predictions):
        if abs(la - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


# set up variational classifier
def variational_classifier(var, angles, mitigate):
    weights = var[0]
    bias = var[1]
    return circuit(program, weights, angles, mitigate) + bias


# cost function: square loss
def cost(weights, features, labels, mitigate):
    predictions = [
        variational_classifier(weights, f, mitigate) for f in features
    ]
    return square_loss(labels, predictions)


# Calculate gradient descent using parameter shift method

# Calculate parameter shift term
def parameter_shift_term(weights, a, labels, mitigate, i):
    shifted = a.copy()
    shifted[i] += np.pi / 2
    forward = cost(weights, shifted, labels, mitigate)  # forward evaluation

    shifted[i] -= np.pi / 2
    backward = cost(weights, shifted, labels, mitigate)  # backward evaluation

    return 0.5 * (forward - backward)


# Calculate gradient wrt all parameters
def parameter_shift(weights, a, labels, mitigate):
    gradients = np.zeros([len(a)])

    for i in range(len(a)):
        gradients[i] = parameter_shift_term(weights, a, labels, mitigate, i)
    return gradients


# load and pre-process data
data = np.loadtxt("data/example_dataset.txt")
X1 = data[:, 0:2]
# print("First X sample (original)   :", X[0])

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X1), 1))
X1_pad = np.c_[np.c_[X1, padding], np.zeros((len(X1), 1))]
# print("First X sample (padded)     :", X_pad[0])

# normalize each input
normalization = np.sqrt(np.sum(X1_pad ** 2, -1))
X1_norm = (X1_pad.T / normalization).T
# print("First X sample (normalized) :", X_norm[0])

# angles for state preparation are new features
features = np.array([calc_angles(x) for x in X1_norm])
# print("First features sample       :", features[0])

Y = data[:, -1]

# generalize from the data samples
np.random.seed(0)
num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
normalization = np.sqrt(np.sum(X1 ** 2, -1))
X1_norm = (X1.T / normalization).T
feats_train = features[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = features[index[num_train:]]
Y_val = Y[index[num_train:]]

# use for plotting later
X1_train = X1[index[:num_train]]
X1_val = X1[index[num_train:]]


# test the quantum program
prog = Program()
var_init = (0.01 * np.random.randn(6, 2, 3), 0.0)
weights = var_init[0]
batch_size = 5
mitigation = False
x_test = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])
ang = calc_angles(x_test)
test_prog = stateprep(prog, ang)
unmitigated = executor_fn(test_prog)
print("Unmitigated expectation value :", unmitigated)

# test the quantum program with noise mitigation
mitigation = True
fac = zne.inference.LinearFactory(scale_factors=[1.0, 3.0])
test_zne = execute_with_zne(test_prog, executor_fn, factory=fac)
print("Expectation value with noise mitigation:", test_zne)

batch_index = np.random.randint(0, num_train, (batch_size,))
feats_train_batch = feats_train[batch_index]
Y_train_batch = Y_train[batch_index]
# print("cost", cost(var_init, feats_train_batch, Y_train_batch, mitigation))
# test gradient calculation routine
print(
    "Gradient calculation",
    parameter_shift(var_init, feats_train_batch, Y_train_batch, mitigation),
)


# train the variational classifier without noise mitigation
# initialize variables randomly, w/ fixed seed for traceability
mitigation = False
num_qubits = 2
num_layers = 6
var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
var = var_init
learn_rate = 0.01
batch_size = 5


for it in range(60):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    var -= learn_rate * parameter_shift(
        var, feats_train_batch, Y_train_batch, mitigation
    )

    # Compute predictions on training and validation set
    predictions_train = [
        np.sign(
            variational_classifier(var, f, mitigation)) for f in feats_train
    ]
    predictions_val = [
        np.sign(
            variational_classifier(var, f, mitigation)) for f in feats_val
    ]

    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        """Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f}
        | Acc validation: {:0.7f} "
        """.format(it + 1, cost(
            var, features, Y, mitigation), acc_train, acc_val)
    )
# train variational classifier with noise mitigation
# re-initialize
program = Program()
var = var_init
mitigation = True

for it in range(60):

    # update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    var -= learn_rate * parameter_shift(
        var_init, feats_train_batch, Y_train_batch, mitigation
    )

    # compute predictions and accuracy on training and validation set
    predictions_train = [
        np.sign(
            variational_classifier(var, f, mitigation)) for f in feats_train
    ]
    predictions_val = [
        np.sign(variational_classifier(var, f, mitigation)) for f in feats_val
    ]
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        """Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f}
        | Acc validation: {:0.7f} "
        """.format(
            it + 1, cost(var, features, Y, mitigation), acc_train, acc_val)
    )
# Plot the noise-mitigated results
plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# preprocess gridpoints like data inputs above
padding = 0.3 * np.ones((len(X_grid), 1))
X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid), 1))]
normalization = np.sqrt(np.sum(X_grid ** 2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = np.array(
    [calc_angles(x) for x in X_grid]
)  # angles for state preparation are new features

predictions_grid = [
    variational_classifier(var, f, mitigation) for f in features_grid
]
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
cnt = plt.contourf(
    xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8,
    extend="both"
)

plt.contour(
    xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",),
    linewidths=(0.8,)
)
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
plt.scatter(
    X1_train[:, 0][Y_train == 1],
    X1_train[:, 1][Y_train == 1],
    c="b",
    marker="o",
    edgecolors="k",
    label="class 1 train",
)
plt.scatter(
    X1_val[:, 0][Y_val == 1],
    X1_val[:, 1][Y_val == 1],
    c="b",
    marker="^",
    edgecolors="k",
    label="class 1 validation",
)

plt.scatter(
    X1_train[:, 0][Y_train == -1],
    X1_train[:, 1][Y_train == -1],
    c="r",
    marker="o",
    edgecolors="k",
    label="class -1 train",
)

plt.scatter(
    X1_val[:, 0][Y_val == -1],
    X1_val[:, 1][Y_val == -1],
    c="r",
    marker="^",
    edgecolors="k",
    label="class -1 validation",
)

plt.legend()
plt.show()
