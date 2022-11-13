##################################################################
# Red neuronal dos entradas y dos salidas               #
# Algoritmo de backpropagation y funcion de activacion sigmoidal #
##################################################################

#LIBRERIAS
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

df = pd.read_csv("Datasets/sujeto.data", header=None)
sec1 = df.iloc[0:100, [0,1]].values
sec2 = df.iloc[100:200, [0,1]].values
sec3 = df.iloc[200:300, [0,1]].values
sec4 = df.iloc[300:400, [0,1]].values


X = df.iloc[0:400, [0,1]].values

y = df.iloc[0:400, 2].values
Y = np.zeros([400,2])

for i in range(len(y)):
    if y[i] == 'sec1':
        Y[i,0] = 0
        Y[i,1] = 0
    
    elif y[i] == 'sec2':
        Y[i,0] = 0
        Y[i,1] = 1
    
    elif y[i] == 'sec3':
        Y[i,0] = 1
        Y[i,1] = 0
       
    elif y[i] == 'sec4':
        Y[i,0] = 1
        Y[i,1] = 1


plt.figure(1)
n=20
colors = plt.cm.jet(np.linspace(0,1,n))
markers = ('s', 'x', 'o', '^', 'v')
plt.title('Conjunto de entrenamiento (DATASET)')


plt.scatter(sec1[:,0], sec1[:,1], color=colors[0], marker=markers[0], 
            edgecolor='black', label='sec 1')  
plt.scatter(sec2[:,0], sec2[:,1], color=colors[5], marker=markers[1], 
            edgecolor='black', label='sec 2')
plt.scatter(sec3[:,0], sec3[:,1], color=colors[10], marker=markers[2], 
            edgecolor='black', label='sec 3')
plt.scatter(sec4[:,0], sec4[:,1], color=colors[15], marker=markers[3], 
            edgecolor='black', label='sec 4')
                
                
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
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
s = 2 # Numero de salidas de la red

topology = [e, 4, 8, 4, s] # Numero de neuronas de la red en cada capa

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
for i in range(15000):
    R = train(neural_net, X, Y, cost, lr=0.001)  
    
    
#%% TEST
'''
    Testeo de la red con el conjunto de entrenamientdo y con un conjunto 
de valores aleatorios de entrada.
'''
a = train(neural_net, X, Y, cost, lr=0.1, train=False)

s1 = np.zeros(len(a))
s2 = np.zeros(len(a))
s3 = np.zeros(len(a))
s4 = np.zeros(len(a))

for i , out in enumerate(a):   
        
    if out[0] <= 0.5 and out[1] <= 0.5:
        s1[i] = 1
        s2[i] = 0
        s3[i] = 0
        s4[i] = 0
            
    elif out[0] <= 0.5 and out[1] >= 0.5:
        s1[i] = 0
        s2[i] = 1
        s3[i] = 0
        s4[i] = 0
        
    elif out[0] >= 0.5 and out[1] <= 0.5:
        s1[i] = 0
        s2[i] = 0
        s3[i] = 1
        s4[i] = 0
    
    elif out[0] >= 0.5 and out[1] >= 0.5:
        s1[i] = 0
        s2[i] = 0
        s3[i] = 0
        s4[i] = 1


markers = ['s', 'x', 'o', '^', 'v']
cmap = ListedColormap(colors[:4])

plt.figure(4)
plt.scatter(X[s1==1, 0], X[s1==1, 1], color=colors[0], marker=markers[0], 
            label='sec 1',edgecolor='black')
plt.scatter(X[s2==1, 0], X[s2==1, 1], color=colors[5], marker=markers[1], 
            label='sec 2', edgecolor='black')
plt.scatter(X[s3==1, 0], X[s3==1, 1], color=colors[10], marker=markers[2], 
            label='sec 3',edgecolor='black')
plt.scatter(X[s4==1, 0], X[s4==1, 1], color=colors[15], marker=markers[3], 
            label='sec 4', edgecolor='black')

plt.title('Testeo con conjunto de entrenamiento')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



#%% ---------------------------------------------------------------------------


m0_min, m0_max = X[:,0].min(), X[:,0].max()
m1_min, m1_max = X[:,1].min(), X[:,1].max()

M = np.random.rand(400,2) # 400 valroes para cada una de las 2 entradas.
M[:,0] = M[:,0] * m0_min + (1-M[:,0]) * m0_max
M[:,1] = M[:,1] * m1_min + (1-M[:,1]) * m1_max

a = train(neural_net, M, Y, cost, lr=0.1, train=False)


s1 = np.zeros(len(a))
s2 = np.zeros(len(a))
s3 = np.zeros(len(a))
s4 = np.zeros(len(a))

for i , out in enumerate(a):   
        
    if out[0] <= 0.5 and out[1] <= 0.5:
        s1[i] = 1
        s2[i] = 0
        s3[i] = 0
        s4[i] = 0
            
    elif out[0] <= 0.5 and out[1] >= 0.5:
        s1[i] = 0
        s2[i] = 1
        s3[i] = 0
        s4[i] = 0
        
    elif out[0] >= 0.5 and out[1] <= 0.5:
        s1[i] = 0
        s2[i] = 0
        s3[i] = 1
        s4[i] = 0
    
    elif out[0] >= 0.5 and out[1] >= 0.5:
        s1[i] = 0
        s2[i] = 0
        s3[i] = 0
        s4[i] = 1


Sec_1 = np.array([M[s1==1, 0], M[s1==1, 1]]).T
Sec_2 = np.array([M[s2==1, 0], M[s2==1, 1]]).T
Sec_3 = np.array([M[s3==1, 0], M[s3==1, 1]]).T
Sec_4 = np.array([M[s4==1, 0], M[s4==1, 1]]).T
secciones = [np.copy(Sec_1), np.copy(Sec_2), np.copy(Sec_3), np.copy(Sec_4)]

contorno_sec1=contorno_sec2=contorno_sec3=contorno_sec4 = []
contornos=[contorno_sec1, contorno_sec2, contorno_sec3, contorno_sec4]

eta=5
for i in range(len(secciones)):
    
    X1 = secciones[i][:,0]
    X1.sort()
    
    X2=np.zeros_like(X1)
    for j in range(len(X2)):
        X2[j]=secciones[i][np.where(secciones[i][:,0]==X1[j]), 1]
        
    sec_actual = np.array([X1,X2]).T
    
    n=0
    while n*eta+eta <= len(secciones[i]):
        max_y,min_y = (np.max(sec_actual[n*eta:n*eta+eta,1]), 
                       np.min(sec_actual[n*eta:n*eta+eta,1]))
        
        maximo = np.array([float(X1[np.where(X2==max_y)]), max_y])
        minimo = np.array([float(X1[np.where(X2==min_y)]), min_y])
        
        contornos[i].append(maximo)
        contornos[i].append(minimo)
        
        n+=1
    
    for t in range (len(sec_actual)//15):
        punto_1 = np.array([sec_actual[t,0], sec_actual[t,1]])
        punto_2 = np.array([sec_actual[len(sec_actual)-t-1,0], sec_actual[len(sec_actual)-t-1,1]])
        
        contornos[i].append(punto_1)
        contornos[i].append(punto_2)

C_sec1=C_sec2=C_sec3=C_sec4=np.zeros((len(contorno_sec1),2))        
  
for i in range(len(contorno_sec1)):
    C_sec1[i,0]= contorno_sec1[i][0] 
    C_sec1[i,1]= contorno_sec1[i][1] 
    
    C_sec2[i,0]= contorno_sec2[i][0] 
    C_sec2[i,1]= contorno_sec2[i][1] 
    
    C_sec3[i,0]= contorno_sec3[i][0] 
    C_sec3[i,1]= contorno_sec3[i][1] 
    
    C_sec4[i,0]= contorno_sec4[i][0] 
    C_sec4[i,1]= contorno_sec4[i][1] 


    
        
    
    

         


plt.figure(5)
plt.scatter(Sec_1[:,0], Sec_1[:,1], color=colors[0], marker=markers[0], 
            label='sec 1',edgecolor='black')
plt.scatter(Sec_2[:,0], Sec_2[:,1], color=colors[5], marker=markers[1], 
            label='sec 2', edgecolor='black')
plt.scatter(Sec_3[:,0], Sec_3[:,1], color=colors[10], marker=markers[2], 
            label='sec 3',edgecolor='black')
plt.scatter(Sec_4[:,0], Sec_4[:,1], color=colors[15], marker=markers[3], 
            label='sec 4', edgecolor='black')

plt.title('Testeo con conjunto de entrenamiento')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


resolution=0.07

xx1, xx2 = np.meshgrid(np.arange(m0_min, m0_max, resolution),
                       np.arange(m1_min, m1_max, resolution))

L = np.array([xx1.ravel(), xx2.ravel()]).T

Z = train(neural_net, L, Y, cost, lr=0.1, train=False)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())





