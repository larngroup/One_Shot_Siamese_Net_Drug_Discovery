# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:41:28 2020

@author: luist
"""

import numpy.random as rng
import numpy as np
import keras
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm
from keras.layers import Input, Conv1D, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling1D, Dropout, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy

class Validate_cnn:
    
    def __init__(self,X_treino,Xval):
        self.Xval = Xval
        self.X_treino = X_treino
        self.n_classes,self.n_exemplos,self.w,self.h = X_treino.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def batch_function(self,n,s='train'):
        
        if s == 'train':
            X = self.X_treino
        else:
            X = self.Xval
                   
        n_classes, n_exemplos, w, h = X.shape
    
        """Cria um batch de n pares, metade da mesma classe e a outra metade de diferentes classes para o treino da rede"""
        categorias = rng.choice(n_classes,size=(n,),replace=False)
        pares=[np.zeros((n, w, h,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            categoria = categorias[i]
            index_1 = rng.randint(0,n_exemplos)
            pares[0][i,:,:,:] = X[categoria,index_1].reshape(w,h,1)
            index_2 = rng.randint(0,n_exemplos)
            #pick images of same class for 1st half, different for 2nd
            
            categoria_2 = categoria if i >= n//2 else (categoria + rng.randint(1,n_classes)) % n_classes
            pares[1][i,:,:,:] = X[categoria_2,index_2].reshape(w,h,1)
        
        return pares, targets
    
    
    
    def validate_cnn(self, N, trials, tam, s= 'val'):
        
        pairs_train, targets_train = self.batch_function(tam)
        
        lista1 = pairs_train[0]
        lista2 = pairs_train[1]
    
        pairs_train2 = []
        
        for i in range(len(lista1)):
            seq3=[]
            seq = lista1[i].flatten()
            seq2 = lista2[i].flatten()
    
            for j in seq:
                seq3.append(j)
            for k in seq2:
                seq3.append(k)
            
            pairs_train2.append(np.asarray(seq3))
        
        n_corretos = 0
        pairs2train=np.asarray(pairs_train2).reshape(tam,54*100*2,1)
        targets_train = np.asarray(targets_train).reshape(tam,1)
        print(pairs2train.shape)
        print(targets_train.shape)
        n_timesteps, n_features, n_outputs = pairs2train.shape[1], pairs2train.shape[2], targets_train.shape[1]
        
        cnn_net = self.cnn_model(pairs2train, targets_train,n_timesteps, n_features, n_outputs)
        
        lista_acc=[]
        for n in range(2,N+1):
            for t in range(trials):
            
                print(t)
                pairs_val2=[]
                pairs_val,targets_val = self.one_shot_task(n,s)
                lista11= pairs_val[0]
                lista22 = pairs_val[1]
                
                for i2 in range(len(lista11)):
                
                    seq3=[]
                    seq = lista11[i2].flatten()
                    seq2 = lista22[i2].flatten()
    
                    for j2 in seq:
                        seq3.append(j2)
                    for k2 in seq2:
                        seq3.append(k2)
                        
                    pairs_val2.append(np.asarray(seq3))
                                
                pairs2val=np.asarray(pairs_val2).reshape(n,54*100*2,1)
    
                targets_val = np.asarray(targets_val).reshape(n,1)
    
                print(pairs2val.shape)
                print(targets_val.shape)
    
                val2 = cnn_net.predict(pairs2val)
    
                
                print("Target:",targets_val)
    
                print("Previsão Probabilidade:",val2)
       
                pred=[]
                for i in val2:
                    pred.append(i[0])
                
                if np.argmax(pred) == 0:
                    n_corretos +=1
                    
            percent_correct = (n_corretos / trials)
            
            lista_acc.append(percent_correct)
            n_corretos= 0
        
        print(lista_acc)
        return lista_acc
  

    def cnn_model(self,trainX, trainy,n_timesteps, n_features, n_outputs):
        
        verbose, epochs, batch_size = 1, 10, 50
    
        conv_model = Sequential()
        conv_model.add(Conv1D(32,10,activation='relu', input_shape=(n_timesteps,n_features)))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(64,8,activation='relu', input_shape=(n_timesteps,n_features)))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(128,4,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(256,2,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Dropout(0.5))
        conv_model.add(MaxPooling1D())
        conv_model.add(Flatten())
        conv_model.add(Dense(100,activation="relu"))
        conv_model.add(Dense(n_outputs, activation='sigmoid'))
        
        conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        conv_model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        return conv_model


        
        
        
        
        
        