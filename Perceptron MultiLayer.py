# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:07:36 2020

@author: Rohr
"""


import numpy as np

inp = np.array ([[0,0],
                 [0,1],
                 [1,0],
                 [1,1]])

out = np.array ([[0],[1],[1],[0]])


wgt1 = 2 * np.random.random((2,3)) - 1


wgt2 = 2 * np.random.random((3,1)) - 1
                 
time = 100000
learningRate = .5
momentun = 1

def sigmoid(s):
    
    return 1 / (1 + np.exp(-s))


def sigmoidDer(sig):
    
    return sig * (1 -sig) 


for i in range(time):
    sumWgt1 = np.dot(inp, wgt1)
    hidenL = sigmoid(sumWgt1)
    
    sumWgt2 = np.dot(hidenL, wgt2)
    outL = sigmoid(sumWgt2)
    
    outError = out - outL
    meanAbs = np.mean(np.abs(outError))
    
    print("Erro: " + str(meanAbs))
    
    derivadaOut = sigmoidDer(outL)
    deltaOut = outError * derivadaOut
    
    deltaWgts2 = deltaOut.dot(wgt2.T)
    
    deltaHidden = deltaWgts2 * sigmoidDer(hidenL)
    
    newWgt2 = hidenL.T.dot(deltaOut)
    
    wgt2 = (wgt2 * momentun) + (newWgt2 * learningRate)
    
    newWgt1 = inp.T.dot(deltaHidden)
    
    wgt1 = (wgt1 * momentun) + (newWgt1 * learningRate)
    
    
    
    
    
    
