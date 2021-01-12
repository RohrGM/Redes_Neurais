# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 22:51:03 2021

@author: Rohr
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy as np

rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
base.addSample(((0,0)), (0))
base.addSample(((0,1)), (1))
base.addSample(((1,0)), (1))
base.addSample(((1,1)), (0))

learning = BackpropTrainer(rede, dataset = base, learningrate = 0.1)

for i in range(0, 10000):
    error = learning.train()
    
    if i % 1000 == 0:
        print("Error: ", error)
        
        
print(np.round(rede.activate([0,0])))
print(np.round(rede.activate([1,0])))
print(np.round(rede.activate([0,1])))
print(np.round(rede.activate([1,1])))
