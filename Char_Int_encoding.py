# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:47:29 2020

@author: luist
"""


from __future__ import print_function
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import time
import os
from Siamese_Loader import Load_Siamese
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import heapq
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from molvs import validate_smiles
from tensorflow import keras

class Process_data:
    
    def remove_duplicates(x):
        
        return list(dict.fromkeys(x))

    def smiles_encoder(smiles):
        #nelson: {"I": 1, "7": 2, ".": 3, "l": 4, "8": 5, "(": 6, "[": 7, "2": 8, "C": 9, "O": 10, "+": 11, "F": 12, "9": 13, "S": 14, ")": 15, "M": 16, "4": 17, "-": 18, "N": 19, "1": 20, "3": 21, "]": 22, "B": 23, "r": 24, "#": 25, "P": 26, "=": 27, "H": 28, "a": 29, "5": 30, "6": 31, "g": 32}
        
        d = {'I': 1, '7': 2, '.': 3, 'l': 4, '8': 5, '(': 6, '[': 7, '2': 8, 'C': 9, 'O': 10, '+': 11, 'F': 12, '9': 13, 'S': 14, ')': 15, 'M': 16, '4': 17, '-': 18, 'N': 19, '1': 20, '3': 21, ']': 22, 'B': 23, 'r': 24, '#': 25, 'P': 26, '=': 27, 'H': 28, 'a': 29, '5': 30, '6': 31, 'g': 32,'c':33, 'n':34, 's': 35, '@':36, '/':37, '\\':38, 'A':39,'i':40,'u':41,'d':42,'o':43,'e':44,'Z':45,'K':46,'V':47,'Y':48,'b':49,'T':50,'G':51,'D':52,'y':53,'t':54}
        #,'@': 33,'o':34,'\\':35, '/':36, 'e':37,'A':38,'Z':39,'K':40, '%':41,'0':42,'i':43,'T':44,'c':45,'s':46,'G':47,'d':48,'n':49,'u':50,'V':51,'R':52,'b':53,'L':54
        X = np.zeros((100,54))
    #    print(smiles)
        for i, valor in enumerate(smiles):
            if(d.get(valor) == None):
                print(valor)
    #        if d.get(valor) == None:
    #           print(valor)
            X[i-1, int(d.get(valor))-1] = 1
      
        return X
    
    