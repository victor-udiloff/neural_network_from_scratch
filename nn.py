import numpy as np
import matplotlib.pyplot as plt

def sigmoid(A):
  return 1/(1+np.exp(-1*A))

def dsigmoid(A):
  return sigmoid(A)*(1-sigmoid(A))

def seno(A):
  return np.sin(A)

def logg(A):
  return np.log10(A)

def inv(A):
  return 1/A

def expp(A):
  return np.exp(-1*A)

def foward(a): #usado para testar a rede
  global W1,W2
  x = np.ones((2,a.shape[0]))
  x[1,:] = a
  y1 = np.ones((l+1,a.shape[0]))
  v1 = np.ones((l+1,a.shape[0]))
  v1[1:,:] = np.matmul(W1,x)
  y1 = sigmoid(v1)
  y1[0,:] = np.ones(a.shape[0])
  v2 = np.matmul(W2,y1)
  y2 = sigmoid(v2)
  return y2


# hiperparemetros
eta = 10 # taxa de aprendizado
Ne = 1000 # numero de epocas
Nb = 1 # tamanho do batch
l = 5 # numero de neuronios da camada oculta
W1 = 2*np.ones((l,2)) # inicalização dos pessos entrada para a camada oculta
W2 =2* np.ones((1,l+1)) # inicalização dos pessos da camada oculta para a saida

#treino
def treinar(f1):

  global W1,W2
  W1 = 2*np.ones(W1.shape)
  W2 =2* np.ones(W2.shape)
  e = np.zeros(Ne)
  
  for i in range(0,Ne):
    #forward
    x = np.ones((2,Nb))
    x[1,:]= np.random.rand(Nb) 
    x[0,:]= np.ones(Nb) 

    y1 = np.ones((l+1,Nb))
    v1 = np.ones((l+1,Nb))
    v1[1:,:] = np.matmul(W1,x)
    y1 = sigmoid(v1)
    y1[0,:] = np.ones(Nb)
    v2 = np.matmul(W2,y1)
    y2 = sigmoid(v2)
    e[i] = np.mean(f1(x[1,:])- y2)
    #backward

    delta2 = dsigmoid(v2)*e[i]
    delta1 = np.multiply(dsigmoid(v1[1:,:]),(np.transpose(W2[:,1:])*delta2))
    deltadelta1 = np.matmul(delta1,(np.transpose(x)))
    deltadelta2 = np.matmul(delta2,np.transpose(y1))
    W1 = W1 + eta* deltadelta1
    W2 = W2 + eta* deltadelta2


#teste
#plotar funções
data_teste_seno = np.linspace(0,3.1415/2,1000)
funcao_real_seno = seno(data_teste_seno)
treinar(seno)
funcao_aproximada_seno = foward(data_teste_seno)
fig = plt.figure()
fig1= fig.add_subplot(2,2,1)
fig1.plot(funcao_real_seno)
fig1.plot(funcao_aproximada_seno[0,:])

data_teste_log = np.linspace(1,10,1000)
funcao_real_log = logg(data_teste_log)
treinar(logg)
funcao_aproximada_log = foward(data_teste_log)
fig = plt.figure()
fig2= fig.add_subplot(2,2,2)
fig2.plot(funcao_real_log)
fig2.plot(funcao_aproximada_log[0,:])

data_teste_inv = np.linspace(1,100,1000)
funcao_real_inv = inv(data_teste_inv)
treinar(inv)
funcao_aproximada_inv = foward(data_teste_inv)
fig = plt.figure()
fig2= fig.add_subplot(2,2,3)
fig2.plot(funcao_real_inv)
fig2.plot(funcao_aproximada_inv[0,:])

data_teste_exp = np.linspace(1,10,1000)
funcao_real_exp = expp(data_teste_exp)
treinar(expp)
funcao_aproximada_exp = foward(data_teste_exp)
fig = plt.figure()
fig2= fig.add_subplot(2,2,4)
fig2.plot(funcao_real_exp)
fig2.plot(funcao_aproximada_exp[0,:])
plt.show()