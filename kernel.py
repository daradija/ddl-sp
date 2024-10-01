from drnumba import *
import numpy as np
import math

@cuda.jit
def NumbaNN_predict(weights,data,activation):
	idx=cuda.grid(1)
	if idx>=weights.shape[1]:
		return
	
	for i in range(weights.shape[0]):
		c=np.float16(idx==weights.shape[1]-1)
		for j in range(weights.shape[2]):
			c+=weights[i][idx][j]*data[j]

		if activation[i]==1: # sigmoid
			c=1/(1+math.exp(-c))
		
		cuda.syncthreads()
		data[j]=c
		cuda.syncthreads()

@cpu.jit
def NumbaNN_CPU_predict(weights,data,activation):
	idx=cpu.grid(1)
	if idx>=weights.shape[1]:
		return
	
	for i in range(weights.shape[0]):
		c=np.float16(idx==weights.shape[1]-1)
		for j in range(weights.shape[2]):
			c+=weights[i][idx][j]*data[j]

		if activation[i]==1: # sigmoid
			c=1/(1+math.exp(-c))
		
		cpu.syncthreads()
		data[j]=c
		cpu.syncthreads()
