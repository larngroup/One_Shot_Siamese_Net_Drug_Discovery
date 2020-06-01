# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:22:30 2019

@author: luist
"""
import numpy as np
from Siamese_Loader import Siamese_Loader
import matplotlib.pyplot as plt

def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways,n_trials,loader):
    """Returns accuracy of one shot """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = loader.make_oneshot_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials


def test_pipeline(data_treino,data_teste,model):
    loader = Siamese_Loader(data_treino,data_teste)
    
    ways = np.arange(1, 30, 2)
    resume =  False
    val_accs, train_accs,nn_accs = [], [], []
    trials = 450
    for N in ways:
        val_accs.append(loader.test_oneshot(model, N,trials, "val", verbose=True))
        train_accs.append(loader.test_oneshot(model, N,trials, "train", verbose=True))
        nn_accs.append(test_nn_accuracy(N,trials, loader))
        
    #plot the accuracy vs num categories for each
    plt.plot(ways, val_accs, "m")
    plt.plot(ways, train_accs, "y")
    plt.plot(ways, nn_accs, "c")
    
    plt.plot(ways,100.0/ways,"r")
    plt.show()
