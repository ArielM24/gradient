import numpy as np
import random


def read_data(file):
	f = open(file)
	data = f.read().split()[1:]
	f.close()

	mat = []
	for row in data:
		mat.append(np.insert(np.array(row.split(",")[2:]).astype(np.float),0,1))
	return mat

def shuffle(data):
	random.shuffle(data,random.random)
	#print(type(aux))
	lim = int(len(data)*0.7)
	train = data[:lim]
	test = data[lim:]
	return [train,test]

def mat_x(mat):
	xs = []
	for row in mat:
		xs.append((row[:-1]))
	return np.array(np.matrix(xs).T)

def mat_y(mat):
	y = []
	for row in mat:
		y.append(row[-1])
	return np.array(y)


def f_scaling(mat_x):
	sca = []
	sca.append(mat_x[0])
	for i in range(1,len(mat_x)):
		x = mat_x[i]
		miu = np.average(x)
		si = np.std(x)
		#print(i,si,type(si))
		x = (x - miu) / si
		sca.append(x)
	return np.array(sca)

def gradient(x,y):
	n = len(x)
	print(n)


if __name__ == '__main__':
	mat = read_data("prueba.txt")
	train, test = shuffle(mat)
	x = mat_x(train)
	y = mat_y(train)
	#print(len(train))
	#print()
	#print(y)
	xf = f_scaling(x)
	#print(len(xf))
	gradient(x,y)
