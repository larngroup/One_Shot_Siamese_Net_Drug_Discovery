# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:04:53 2019

@author: luist
"""

from __future__ import print_function
import csv
import heapq
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from molvs import validate_smiles
from sklearn.model_selection import train_test_split
import numpy as np


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
        print(d)
        if len(d) > 5 and c>0 and c<=1000:
            drug_names.append(d[12])
            drug_group.append(d[2])
            drug_smiles.append(d[13])
        c+=1  
    
#    for drug in data:
#        print(drug_names[0:3])
#        print(drug_smiles[0:3])
#        print(drug_group[0:3])
#    
    
    drug_mol=[]
    #aplicando validate_smiles(drug_smiles[1406]) verifica-se que a molecula e invalida
    #remover todas as invalidas
    for i in drug_smiles:
        if validate_smiles(i) != []:
            drug_smiles.pop(drug_smiles.index(i))
            
    
    drug_smiles = list(filter(lambda a: len(a) <= 100 and len(a) > 0, drug_smiles))    
#    print(validate_smiles(drug_smiles[1407]))
#    print(len(drug_smiles))
    
    l=[]
    for j in drug_smiles:
        drug_mol.append(Chem.MolFromSmiles(j))
        l.append(len(j))
       
#    print(l)
    
    fps = [FingerprintMols.FingerprintMol(x) for x in drug_mol]
    
    prob = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    
    n=5
    count = 0
   
    # experimentar diferentes métricas de semelhança 
    # Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    
    Prob = []
    Data_Final=[]
    Data_Final1=[]
    data_prov=[]
#    print(fps)
    for i in fps:
        probs=[]
        for j in fps:
            probs.append(DataStructs.FingerprintSimilarity(i,j))
     
        indexes = heapq.nlargest(len(probs), range(len(probs)), probs.__getitem__)
  
#        fps = [i for j, i in enumerate(fps) if j not in indexes]
        Data1=[drug_smiles[k] for k in indexes]
        Data=[]
        Data2=[]
        c = 0
        for d in Data1:
            if d not in data_prov and c <=n:
                Data2.append(d)
                data_prov.append(d)
            c+=1
        
        for d2 in Data2:
            Data.append(smiles_encoder(d2))
            

        if (len(Data2)==n):
            Data_Final.append(Data)
            Data_Final1.append(Data2)
#        Data= [smiles_encoder(drug_smiles[k]) for k in indexes]
#        print(Data1)
#        Prob.append([probs[k] for k in indexes])
#        count = 0
#        for i in Data1:
#            count+=1
#            if(i not in data_prov and count == 1):
#                data_prov.append(i)
#                Data_Final.append(Data)
#                Data_Final1.append(Data1)
    
#        for i in Data1:
#            count+=1
#            if(i not in data_prov and count == 1):
#                data_prov.append(i)
#                Data_Final.append(Data)
#                Data_Final1.append(Data1)
#                print(i)
#
 
    #Verificar se todos os items são diferentes
    data = []
    for i in Data_Final1:
        for j in i:
            data.append(j)
    
  
    print (len(data) != len(set(data)))
        
            
#    print(Data_Final1)        
    
#    print(Data_Final[0])

    data_treino, data_teste = train_test_split([i for i in Data_Final], test_size=0.25, random_state=1)
    
    data_treino = np.asarray(data_treino)
    data_teste = np.asarray(data_teste)
    
    print(data_treino.shape)
    print(data_teste.shape)
    
    
    


    

