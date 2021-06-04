# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:28:49 2021

@author: misty

Example of variational problem made trainable wtih noise mitigation toolkit Mitiq

Variational classifier problem based on PennyLane variational classifier tutorial: 
https://pennylane.ai/qml/demos/tutorial_variational_classifier.html, 
PennyLane quantum gradient tutorial: https://pennylane.ai/qml/demos/tutorial_backprop.html, 
and Mitiq setup example: 
https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html#general-workflow-front-ends-backends-and-executors
"""
from pyquil import get_qc, Program
from pyquil.gates import RY, RZ, X, MEASURE, CNOT
from mitiq import zne, execute_with_zne
from mitiq.mitiq_pyquil.compiler import basic_compile
from mitiq.mitiq_pyquil.pyquil_utils import ground_state_expectation
from functools import partial 
import numpy as np
import matplotlib.pyplot as plt

# initialize quantum device
qpu = get_qc("2q-noisy-qvm")

# set up quantum circuit
def execute(program: Program, a, weights, shots: int = 100) -> float:
    p = Program()

    # add main body program
    p += RY(angle=a[0], qubit=0)
    p += CNOT(0, 1)
    p += RY(angle=a[1], qubit=1)
    p += CNOT(0, 1)
    p += RY(angle=a[2], qubit=1)
    p += X(qubit=0)
    p += CNOT(0, 1)
    p += RY(angle=a[3], qubit=1)
    p += CNOT(0, 1)
    p += RY(angle=a[4], qubit=1)
    p += X(qubit=0)
    # Apply rotation based on weights
    for W in weights:
           p += RZ(W[0, 0], 0)
           p += RY(W[0, 1], 0)
           p += RZ(W[0, 2], 0)
   
           p += RZ(W[1, 0], 1)
           p += RY(W[1, 1], 1)
           p += RZ(W[1, 2], 1)
           p += CNOT(0, 1)
    
    p += program.copy()

    # add memory declaration
    qubits = p.get_qubits()
    ro = p.declare("ro", "BIT", len(qubits))

    # add measurements
    for idx, q in enumerate(qubits):
        p += MEASURE(q, ro[idx])

    # add numshots
    p.wrap_in_numshots_loop(shots)

    # nativize the circuit
    p = basic_compile(p)

    # compile the circuit
    b = qpu.compiler.native_quil_to_executable(p)

    # run the circuit, collect bitstrings
    qpu.reset()
    results = qpu.run(b)
    
    # compute ground state expectation value
    return ground_state_expectation(results)


# Translate input x into rotation angles for state preparation
def calc_angles(x):
    
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))
   
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# Define loss function as a standard square loss function
def square_loss(labels, predictions):
     loss = 0
     for l,p in zip(labels, predictions):
         loss = loss + (l - p) ** 2
     loss = loss / len(labels)
     return loss

# Define the accuracy given target labels and model predictions, to monitor
# how many inputs the classifier predicted correctly
def accuracy(labels, predictions):
    
     loss = 0
     for l,p in zip(labels, predictions):
         if abs(l - p) < 1e-5:
             loss = loss + 1
     loss = loss / len(labels)
            
     return loss

# set up variational classifier
def variational_classifier(program: Program, var, angles):
      weights = var[0]
      bias = var[1]
    
      return execute(program, angles, weights) + bias 
  
# variational classifier with noise mitigation
def var_classifier_mit(program: Program, var, angles):
      weights = var[0]
      bias = var[1]
    
      return execute_with_zne(program, partial(angles, weights)) + bias 

# cost function: square loss
def cost(program: Program, var, features, labels):
     predictions = [variational_classifier(program, var, f) for f in features]
     return square_loss(labels, predictions)


# Calculate gradient descent using parameter shift method

# Calculate parameter shift term
def parameter_shift_term(program: Program, a, i):
    shifted = a.copy()
    shifted[i] += np.pi/2
    forward = execute(program, shifted, weights)  # forward evaluation

    shifted[i] -= np.pi/2
    backward = execute(program, shifted, weights) # backward evaluation

    return 0.5 * (forward - backward)    
 
# Calculate gradient wrt all parameters
def parameter_shift(program: Program, a):
    gradients = np.zeros([len(a)])

    for i in range(len(a)):
        gradients[i] = parameter_shift_term(program, a, i)

    return gradients


# load and pre-process data
data = np.loadtxt("var_classifier_mitiq/data/example_dataset.txt")
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
    
x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])
ang = calc_angles(x)
unmitigated = execute(prog, ang, weights)
print("Unmitigated expectation value :",         unmitigated)

# test the quantum program with noise mitigation 
fac = zne.inference.LinearFactory(scale_factors = [1.0, 3.0]) 
test_zne = execute_with_zne(prog, partial(execute, ang, weights), factory = fac)
print("Expectation value with noise mitigation:", test_zne)


# test gradient calculation routine
print("Gradient calculation", parameter_shift(prog, ang))


# train the variational classifier without noise mitigation
# initialize variables randomly, w/ fixed seed for traceability
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

    var += learn_rate*parameter_shift(prog, feats_train_batch)
   
    # Compute predictions on training and validation set
    predictions_train = [np.sign(variational_classifier(prog, var, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(prog, var, f)) for f in feats_val]


    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
    "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
    "".format(it + 1, cost(prog, var, features, Y), acc_train, acc_val)
)


# train variational classifier with noise mitigation
# re-initialize
var = var_init
for it in range(60):
    
    # update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    var += learn_rate*parameter_shift(prog, feats_train_batch)
   
    # compute predictions and accuracy on training and validation set
    predictions_train = [np.sign(var_classifier_mit(prog, var, f)) for f in feats_train]
    predictions_val = [np.sign(var_classifier_mit(prog, var, f)) for f in feats_val]
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
    "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
    "".format(it + 1, cost(prog, var, features, Y), acc_train, acc_val)
)


# Plot the noise-mitigated results
plt.figure()
cm = plt.cm.RdBu 

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))
X_grid = [np.array([x,y]) for x,y in zip(xx.flatten(), yy.flatten())]

# preprocess gridpoints like data inputs above
padding = 0.3 * np.ones((len(X_grid),1))
X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid),1))]
normalization = np.sqrt(np.sum(X_grid ** 2, -1))
X_grid = (X_grid.T/normalization).T # normalize each input
features_grid = np.array(
    [calc_angles(x) for x in X_grid]
) # angles for state preparation are new features

predictions_grid = [var_classifier_mit(prog, var, f) for f in features_grid]
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
cnt = plt.contourf(
    xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both"
)

plt.contour(
    xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,)
)
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
plt.scatter(
    X1_train[:, 0][Y_train == 1],
    X1_train[:, 1][Y_train == 1],
    c="b",
    marker="o",
    edgecolors="k",
    label="class 1 train"
)
plt.scatter(
    X1_val[:, 0][Y_val == 1],
    X1_val[:, 1][Y_val == 1],
    c="b",
    marker="^",
    edgecolors="k",
    label="class 1 validation"
)

plt.scatter(
    X1_train[:, 0][Y_train == -1],
    X1_train[:, 1][Y_train == -1],
    c="r",
    marker="o",
    edgecolors="k",
    label="class -1 train"
)

plt.scatter(
    X1_val[:, 0][Y_val == -1],
    X1_val[:, 1][Y_val == -1],
    c="r",
    marker="^",
    edgecolors="k",
    label="class -1 validation"
)

plt.legend()
plt.show()
