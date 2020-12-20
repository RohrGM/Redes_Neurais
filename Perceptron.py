# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:21:09 2020

@author: Rohr
"""

import numpy as np

values_input = np.array([[0,0],[0,1],[1,0],[1,1]])
#values_output = np.array([0, 0, 0, 1])
#values_output = np.array([0, 1, 1, 1])
values_output = np.array([1, 0, 0, 1])

weights = np.array([.0, .0])
learning_rate = .1

def step_func (value_sum):
    if value_sum >= 1:
        return 1
    return 0

def cal_output(inp):
    s = inp.dot(weights)
    
    return step_func(s)

def learning():
    
    total_error = 1
    while total_error != 0:
        total_error = 0
        for i in range(len(values_output)):
            otpt = cal_output(np.array(values_input[i]))
            error = abs(values_output[i] - otpt)
            total_error += error
            
            if error > 0:
            
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learning_rate * values_input[i][j] * error)
                    print("Update Weight: " + str(weights[j]))
                    
        print("Total error: " + str(total_error))
                
                
learning()
    


