#####################################################################
# RED NEURONAL de dos entradas y una salida                         #
# Algoritmo de backpropagation y funcion de activacion sigmoidal    #
#####################################################################

#Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

#%% Conjunto de entrenamiento (DATASET)

'''
    X contiene los valores en cm del sepalo (X[:,0])  y el petalo (X[:,1]) de 
flores iris setosa y versicolor.

    Las posiciones donde Y contiene un 0 son las mismas posiciones de fila que 
en X corresponden a valores de iris-setosa y 1 para las posiciones que 
corresponder a iris-versicolor.
'''

df = pd.read_csv("Datasets/iris.data", header=None)
X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
Y = np.where(y == 'Iris-setosa', 0, 1)
Y = np.array([Y]).T

plt.figure(1)
plt.title('Conjunto de entrenamiento (DATASET)')
plt.scatter(X[:,0], X[:,1])
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.tight_layout()


#%% CLASE DE LA CAPA DE RED

class neural_layer():
    '''
        Clase para guardar los pesos y la funcion de activacion de las 
    neuronas en una capa de la red.
    '''
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)      * 2 - 1
        self.w = np.random.rand(n_conn, n_neur) * 2 - 1


#%% FUNCION DE ACTIVACION

'''
    Funcion de activacion: Sigmoide
    Derivada de la funcion sigmoide
'''

sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))
        
# relu = lambda x: np.maximum(0, x)  # otra funcion de activacion


_x = np.linspace(-5, 5, 100)
plt.figure(2)
plt.title('Funcion sigmoide')
plt.plot(_x , sigm[0](_x))
plt.tight_layout()
plt.figure(3)
plt.title('Funcion sigmoide derivada')
plt.plot(_x , sigm[1](_x))
plt.tight_layout()


#%% RED NEURONAL (CAPAS OCULTAS)

e = 2 # Numero de entradas de la red

topology = [e, 4, 2, 1] # Numero de neuronas de la red en cada capa

def create_nn(topology, act_f):
    '''
        Función que crea una lista que contiene una clase: neural_layer para
    cada capa de la red que contiene los valores de los pesos y la funcion
    de activacion de las neurons de dicha capa.
    '''

    nn = []
    
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))
        
    return nn


neural_net = create_nn(topology, sigm)    


#%% ENTRENAMIENTO DE LA RED

cost = (lambda Yp, Yr: np.mean((Yp-Yr)**2), # Funcion de coste (Ypredicha, Yreal)
        lambda Yp, Yr: (Yp-Yr))             # Derivada de la funcion de coste

        
def train(neural_net, X, Y, cost, lr=0.5, train=True):
    '''
        Funcion de entrenamiento de la red. 
        train=True para entrenamiento.
        train=False para lectura.
    '''
    
    #Forward pass
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        
        z = out[-1][1] @ neural_net[l].w + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        
        out.append((z, a))
        
    if train:
            
        # Backward pass
        deltas = []
                
        for l in reversed(range(0, len(neural_net))):
                
            z = out[l+1][0]
            a = out[l+1][1]
                
            if l == len(neural_net) - 1:
                #calcular delta de ultima capa
                deltas.insert(0, cost[1](a, Y) * neural_net[l].act_f[1](a))
                    
            else:
                # Calcular delta respecto a capa previa
                deltas.insert(0, deltas[0] @ w_aux.T * neural_net[l].act_f[1](a))
                
            w_aux = neural_net[l].w
            
            # Gradient descient
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] * lr
            
    return out[-1][1]
      

# Entrenamiento de la red
for i in range(5000):
    R = train(neural_net, X, Y, cost, lr=0.1)  
    
    
#%% TEST
'''
    Testeo de la red con el conjunto de entrenamientdo y con un conjunto 
de valores aleatorios de entrada.
'''
a = train(neural_net, X, Y, cost, lr=0.1, train=False)

plt.figure(4)

p=np.zeros(len(a))
for i in range(len(a)):   
        
    if a[i]>=0.5:
        p[i]=1
            
    elif a[i]<0.5:
        p[i]=0


plt.scatter(X[p==0,0], X[p==0,1], color='red', marker='o', 
            label='iris-setosa',edgecolor='black')
plt.scatter(X[p==1,0], X[p==1,1], color='blue', marker='x', 
            label='iris-versicolor', edgecolor='black')

plt.title('Testeo con conjunto de entrenamiento')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



# ---------------------------------------------------------------------------


m0_min, m0_max = X[:,0].min(), X[:,0].max()
m1_min, m1_max = X[:,1].min(), X[:,1].max()

M = np.random.rand(300,2) # 300 valroes para cada una de las 2 entradas.
M[:,0] = M[:,0] * m0_min + (1-M[:,0]) * m0_max
M[:,1] = M[:,1] * m1_min + (1-M[:,1]) * m1_max

a = train(neural_net, M, Y, cost, lr=0.1, train=False)


plt.figure(5)
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(Y))])

'''
for i in range(len(a)):   
        
    if a[i]>=0.5:
        plt.scatter(M[i,0], M[i,1], color='blue', marker='x')
            
    elif a[i]<0.5:
        plt.scatter(M[i,0], M[i,1], color='red', marker='o')
'''

# '''
p=np.zeros(len(a))
for i in range(len(a)):   
        
    if a[i]>=0.5:
        p[i]=1
            
    elif a[i]<0.5:
        p[i]=0


plt.scatter(M[p==0,0], M[p==0,1], color=cmap(0), marker='o', 
            label='iris-setosa', edgecolor='black')
plt.scatter(M[p==1,0], M[p==1,1], color=cmap(1), marker='x',
            label='iris-versicolor', edgecolor='black')

# '''


plt.title('Testeo con conjunto de valores aleatorio')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()


resolution=0.02

xx1, xx2 = np.meshgrid(np.arange(m0_min, m0_max, resolution),
                       np.arange(m1_min, m1_max, resolution))

R = np.array([xx1.ravel(), xx2.ravel()]).T

Z = train(neural_net, R, Y, cost, lr=0.1, train=False)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())






