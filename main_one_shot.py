# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:22:44 2019

@author: luist
"""
from __future__ import print_function
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras import backend as K
import numpy as np
import time
from Validate_cnn import Validate_cnn
from Validate_models import Validate_models
from Siamese_Loader import Load_Siamese
from Siamese_Model import Siamese_Net
from Process_Data import Process_Data
from Validate_one_shot import Validate_one_shot
from Validate_knn import Validate_knn
from Validate_random import Validate_random
from sklearn.model_selection import train_test_split
import csv
import heapq
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from molvs import validate_smiles
from Char_Int_encoding import Process_data

def pesos(shape, name=None):

    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
#//TODO: figure out how to initialize layer biases in keras.
def bias(shape, name=None):
 
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def model():
    
    input_shape = (100,54,1)
    
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    print(left_input)
    
    # Convolutional Neural Network
    """ -> convolução com 64 filtros de 10x10;
        -> convolução com filtros ReLU (se g(x) > 0 return value else return 0, com g a função de ativação) - ReLU activation to make all negative value to zero;
        -> convolução com uma max-pooling layer (escolhe o valor mais elevado em sub-matrizes nxn bem definidas) - Pooling layer is used to reduce the spatial volume of input image after convolution;
        -> convolução com 128 filtros de 7x7 + ReLU + maxpool;
        -> convolução com 128 filtros de 2x2 + ReLU + maxpool;

        -> convolução com 256 filtros 2x2;
        -> dropout layer - droprate= 0.5
        -> classificação numa fully connected layer de 1024 unidades (neurónios)"""

    #build convnet for each of the siamese twins
                                 
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_initializer=pesos,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',input_shape=input_shape,
                       kernel_initializer=pesos,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(2,2),activation='relu',
                       kernel_regularizer=l2(2e-4),kernel_initializer=pesos,bias_initializer=bias))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(2,2),activation='relu',kernel_initializer=pesos,kernel_regularizer=l2(2e-4),bias_initializer=bias))
    convnet.add(MaxPooling2D())
    convnet.add(Dropout(0.5))
    convnet.add(Flatten())
    convnet.add(Dense(1024,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=pesos,bias_initializer=bias))
    #encode each of the two inputs into a vector with the convnet
    left_output = convnet(left_input)
    right_output = convnet(right_input)
    
    
    # Layer that computes the absolute difference between the output feature vectors
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1 = L1_layer([left_output, right_output])
    
    # Dense Layer with a sigmoid function that returns the generated similarity score
    # output between 0 and 1
    pred = Dense(1,activation='sigmoid',bias_initializer=bias)(L1)
    
    # Connect the inputs with the outputs
    net = Model(inputs=[left_input,right_input],outputs=pred)

    return net


if __name__=='__main__':
    
    csv_file ='tox21.csv'
    txt_file ='sl.txt'
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write("$$$".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
    
    with open('sl.txt', 'r') as myfile:
      data = myfile.read()
    
    data = data.split("\n")
    
    D=[]
    drug_names = []
    drug_smiles = []
    drug_group = []
    
    c = 0
    for i in data:
        d = i.split("$$$")
        D.append(d)
#        print(d)
        if len(d) > 5 and c>0:
            drug_names.append(d[12])
            drug_group.append(d[2])
            drug_smiles.append(d[13])
        c+=1  
    

    drug_mol=[]
    
    # Process the drug SMILEs Strings - selection of valid SMILEs and SMILEs grouping
   
    for i in drug_smiles:
        if validate_smiles(i) != []:
            drug_smiles.pop(drug_smiles.index(i))
        
    drug_smiles = list(filter(lambda a: len(a) <= 100 and len(a) > 0, drug_smiles))   

    l=[]
    for j in drug_smiles:
        drug_mol.append(Chem.MolFromSmiles(j))
        l.append(len(j))
       
    
    fps = [FingerprintMols.FingerprintMol(x) for x in drug_mol]
    
    prob = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    
    n=10
    count = 0
   
    # Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    
    Prob = []
    Data_Final=[]
    Data_Final1=[]
    data_prov=[]
    for i in fps:
        probs = [DataStructs.FingerprintSimilarity(i,j) for j in fps]
        indexes = heapq.nlargest(n, range(len(probs)), probs.__getitem__)
#        fps = [i for j, i in enumerate(fps) if j not in indexes]
        Data1=[drug_smiles[k] for k in indexes]
        Data= [Process_data.smiles_encoder(drug_smiles[k]) for k in indexes]
#        print(Data1)
        Prob.append([probs[k] for k in indexes])
        count = 0
        for i in Data1:
            count+=1
            if(i not in data_prov and count == 1):
                data_prov.append(i)
                Data_Final.append(Data)
                Data_Final1.append(Data1)
                
    data = []
    for i in Data_Final1:
        for j in i:
            data.append(j)
    
  
    print (len(data) != len(set(data)))
    
    # separate in train and test sets
    
    data_treino, data_teste = train_test_split([i for i in Data_Final], test_size=0.25, random_state=1)
    
    data_treino = np.asarray(data_treino)
    data_teste = np.asarray(data_teste)
    
    print(len(data_treino))
    print(len(data_teste))
    siamese_net = model()
    
    # Hyper parameters 
    evaluate_every = 100 
    loss_every = 1000
    batch_size = 50
    n_iter = 10000
    n_val = 500 
    best_val = -1
    best_train = -1
    best_val_knn = -1
    best_val_random = -1
    N = 2 #3,4,5,7,10
    
#    model_path = './weights/'
    
    loader = Load_Siamese(data_treino,data_teste)
    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    opt = Adam(lr = 0.0001)
    siamese_net.compile(loss="binary_crossentropy",optimizer=opt)
    
    # Uncomment the following commands to run the models
#    Validate_one_shot.validate(n_iter,loader, time, siamese_net,batch_size,t_start,n_val, evaluate_every, loss_every, N);
#    Validate_knn.validate_knn(n_iter,loader, time, siamese_net,batch_size,t_start,n_val,best_val, evaluate_every, loss_every, N);
#    Validate_random.validate_random(n_iter,loader, time, siamese_net,batch_size,t_start,n_val,best_val, evaluate_every, loss_every, N);
    
#    loader.validate_cnn(2, 100,len(data_treino))
#    loader.validate_models(3, 100,len(data_treino),'RF')
#    loader.validate_models(3, 100,len(data_treino),'SVM')
#    loader.validate_models(3, 100,len(data_treino),'MLP')