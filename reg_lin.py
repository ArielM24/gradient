import numpy as np
import random


def read_data(file):
	f = open(file)
	data = f.read().split()[1:]
	f.close()
	mat = []
	for row in data:
		mat.append(np.array(row.split(",")[2:]).astype(np.float))
	return mat

def shuffle(data):
	random.shuffle(data,random.random)
	lim = int(len(data)*0.7)
	train = data[:lim]
	test = data[lim:]
	return [train,test]

def mat_x(mat):
	xs = []
	for row in mat:
		xs.append(np.insert(row[1:],0,1))
	return np.array(xs)

def mat_y(mat):
	y = []
	for row in mat:
		y.append(row[0])
	return np.array(y)


def f_scaling(mat_x):
	sca = []
	for i in range(len(mat_x)):
		x = mat_x[i]
		miu = np.average(x)
		si = np.std(x)
		x = (x - miu) / si
		sca.append(x)
	return np.array(sca)

def cost(x,y,theta,m):
	su = 0
	for i in range(m):
		su = su + ((hyp(x[i],theta)-y[i])**2)
	return (1/(2*m))*su

def hyp(x,theta):
	return x.dot(theta)

def suma(x,theta,y,m,j):
	su = 0
	for i in range(m):
		su = su + (hyp(x[i],theta) - y[i])*x[i][j]
	return su

def gradient(x,y,a = 0.1,init = 0,ite = 1000,show = 50):
	m = len(x)
	n = len(x[0])
	theta = []
	aux = []
	for i in range(n):
		theta.append(init)
		aux.append(init)

	for c in range(ite):
		for j in range(n):
			aux[j] = aux[j] - ((a/m) * suma(x,theta,y,m,j))
		for j in range(n):
			theta[j] = aux[j]
		if (c + 1)%show == 0:
			print("cost at iteration",c+1,cost(x,y,theta,m))
	return theta


if __name__ == '__main__':
	mat = read_data("data.txt")
	train, test = shuffle(mat)
	x_train = mat_x(train)
	y_train = mat_y(train)
	x_train = f_scaling(x_train)
	theta = gradient(x_train,y_train)
	print(theta)

	x_test = mat_x(test)
	y_test = mat_y(test)
	f = open("test.txt","w")
	for i in range(len(x_test)):
		h = hyp(x_test[i],theta)
		f.write("hypothesis: "+ str(h)+" correct: "+str(y_train[i])+"\n")
	f.close()
