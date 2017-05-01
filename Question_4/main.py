from __future__ import print_function

import sys,os,time
from copy import copy,deepcopy

import matplotlib
from matplotlib import pyplot as plt

import numpy as np

def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0

def plot(i1,i2,y,weights=None):
	fig,ax = plt.subplots()
	ax.set_xlabel("i1")
	ax.set_ylabel("i2")

	if weights!=None:
		map_min=0.0
		map_max=1.1

		y_res=0.001
		x_res=0.001

		ys=np.arange(map_min,map_max,y_res)
		xs=np.arange(map_min,map_max,x_res)
		zs=[]
		for cur_y in np.arange(map_min,map_max,y_res):
			cur_zs=[]
			for cur_x in np.arange(map_min,map_max,x_res):
				zs.append(predict([1.0,cur_x,cur_y],weights))
		xs,ys=np.meshgrid(xs,ys)
		zs=np.array(zs)
		zs = zs.reshape(xs.shape)
		cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

	c1_data=[[],[]]
	c0_data=[[],[]]
	for i in range(len(i1)):
		cur_i1 = i1[i]
		cur_i2 = i2[i]
		cur_y  = y[i]
		if cur_y==1:
			c1_data[0].append(cur_i1)
			c1_data[1].append(cur_i2)
		else:
			c0_data[0].append(cur_i1)
			c0_data[1].append(cur_i2)

	plt.xticks(np.arange(0.0,1.1,0.1))
	plt.yticks(np.arange(0.0,1.1,0.1))
	plt.xlim(0,1.05)
	plt.ylim(0,1.05)

	c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
	c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

	plt.legend(fontsize=10,loc=1)
	plt.show()

def accuracy(i0,i1,i2,y,weights):
	num_correct=0.0
	predictions=[]
	for i in range(len(i0)):
		inputs=[i0[i],i1[i],i2[i]]
		prediction = predict(inputs,weights)
		predictions.append(prediction)
		if y[i]==prediction: num_correct+=1.0
	print("Predictions:",predictions)
	return num_correct/float(len(i0))

def train_weights(i0,i1,i2,y,weights,nb_epoch,l_rate,do_plot=False,stop_early=True):
	for epoch in range(nb_epoch):
		print("\nEpoch %d"%epoch)
		print("Weights: ",weights)
		cur_acc = accuracy(i0,i1,i2,y,weights)
		print("Accuracy: ",cur_acc)
		if cur_acc==1.0 and stop_early: break
		if do_plot: plot(i1,i2,y,weights)
		for i in range(len(i0)):
			inputs=[i0[i],i1[i],i2[i]] # first is bias input
			prediction = predict(inputs,weights) # predict 1.0 or 0.0
			error = y[i]-prediction  
			for j in range(len(weights)):
				weights[j] = weights[j]+(l_rate*error*inputs[j])
	plot(i1,i2,y,weights)
	return weights 

def main():

	i0 = [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00] # constant bias inputs 
	i1 = [0.08,0.10,0.26,0.35,0.45,0.60,0.70,0.92] # x axis of plot
	i2 = [0.72,1.00,0.58,0.95,0.15,0.30,0.65,0.45] # y axis of plot

	y = [1.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0] # 1.0 for Class1, 0.0 for Class-1

	weights = [0.20,1.00,-1.00] # initial weights specified in problem
	
	train_weights(i0,i1,i2,y,weights=weights,nb_epoch=50,l_rate=1.0,do_plot=True)

if __name__ == '__main__':
	main()