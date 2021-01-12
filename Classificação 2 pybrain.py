# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 01:03:04 2021

@author: Rohr
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
base.addSample(((8,4)), (10))
base.addSample(((7,1)), (5))
base.addSample(((10,0)), (1))
base.addSample(((0,4)), (1))



learning = BackpropTrainer(rede, dataset = base, learningrate = 0.01)

for i in range(0, 10000):
    print(learning.train())   
    
    
while True:

    dormiu = float(input('Dormiu\n'))
    estudou = float(input('Estudou\n'))
    
    previsao = rede.activate((dormiu, estudou))
    
    print("Previsão da nota: ", previsao)


        
