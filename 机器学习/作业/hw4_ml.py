import numpy as np
import matplotlib.pyplot as plt

W = np.mat([[0.8,-0.1],[-0.12,0.8]])
U = np.mat([[2,-1],[1,1]])
V = np.mat([[0.5,1]])
b = np.mat([[0.2],[-0.1]])
c = 0.25
t = np.arange(1,10,0.01)
n = len(t)
x = np.row_stack([np.sin(0.2*np.pi*t), np.cos(0.5*np.pi*t)])
x = np.mat(x)
h = np.zeros([2,n])
h = np.mat(h)
y = np.zeros(n)
# x0 = x[:,0].reshape(-1,1)
# h0 = h[:,0].reshape(-1,1)
h[:,0] = np.tanh(b+np.dot(U,x[:,0]))
y[0] = c + np.dot(V,h[:,0])
for i in range(1,n):
    # xi = x[:,i].reshape(-1,1)
    # hi_1 = h[:,i-1].reshape(-1,1)
    h[:,i] = np.tanh(b+np.dot(W,h[:,i-1])+np.dot(U,x[:,i]))
    y[i] = c+np.dot(V,h[:,i])

plt.figure()
plt.plot(t,y)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

x = np.mat([[1,-1,1],[2,0,-1]])
y = np.array([-1,1,2])
n=3
h=np.mat(np.zeros([2,n]))
yhat=np.zeros(n)
tol=1e-8

while np.sqrt(np.sum(np.square(yhat-y)))>1e-8:
    h[:,0] = np.tanh(b+np.dot(U,x[:,0]))
    yhat[0] = c+np.dot(V,h[:,0])
    for i in range(1,n):
        h[:,i] = np.tanh(b+np.dot(W,h[:,i-1])+np.dot(U,x[:,i]))
        yhat[i] = c+np.dot(V,h[:,i])
    diffo = yhat-y
    diffh = x
    diffh[:,n-1] = np.dot(V.T, diffo[n-1])    
    for i in range(n-2,-1,-1):
        diffh[:,i] = np.dot(W.T, np.dot(np.diag(1-h[:,i+1])*h[:,i+1],diffh[i+1])+np.dot(V.T,diffo[i]))
    diffc = np.sum(diffo)
    diffV = np.dot(diffo,h.T)
    diffb = np.zeros(2)
    for i in range(n):
        diffb = diffb + np.dot(np.diag(1-np.square(h[:,i])),diffh[:,i])
    diffW = np.mat(np.zeros([2,2]))
    for i in range(1,n):
        diffW = diffW + np.dot(np.dot(np.diag(1-np.square(h[:,i])),diffh[:,i]),diffh[:,i-1].T)
    diffU = np.mat(np.zeros([2,2]))
    for i in range(n):
        diffU = diffU + np.dot(np.dot(np.diag(1-np.square(h[:,i])),diffh[:,i]),x[:,i].T)
    
    c = c-0.01*diffc
    V = c-0.01*diffV
    W = c-0.01*diffW
    U = c-0.01*diffU
    b = c-0.01*diffb