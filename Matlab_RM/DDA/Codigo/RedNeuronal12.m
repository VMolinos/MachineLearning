%% Red Neuronal
close all;clear;clc;

%% Entradas
ncoef = 3;
limite = 10*(ncoef+1);
ngrabaciones = 50;

% Entradas de Víctor 
xdigV1 = load('Coeficientes/Uno/uno.mat');
xdigV1 = xdigV1.CepsCoef(11:limite,1:ngrabaciones);
xdigV2 = load('Coeficientes/Dos/dos.mat');
xdigV2 = xdigV2.CepsCoef(11:limite,1:ngrabaciones);

x = horzcat(xdigV1,xdigV2);
[m,n] = size(x);

% Representación de coeficientes
idx = 1;
figure(idx)
mesh(x); title('Coeficientes')
xlabel('grabación');ylabel('muestras/tramas');zlabel('Valor');


%% Salida

y0 = zeros(1,n/2);
y1 = ones(1,n/2);
y  = horzcat(y0,y1);

idx = idx+1;
figure(idx)
stem(y); title('Salida')


%% Red
%PR = minmax(x); % Valor mínimo y máximo de cada variable. Por filas.
S = 30; % Topología de la red. Cantidad de neuronas en cada layer.
TF = {'logsig'}; % Función de activación en cada layer
%TF = {'tansig','tansig','logsig'}; % Función de activación en cada layer

%net = newff(PR,S,TF);
net = newff(x,y,S);
net.trainParam.epochs=1000; % Cantidad máxima de iteracciones.
[net_tra12,~,a,~] = train(net,x,y);
%save('RedEntrenada/net_tra12','net_tra12')
round(a)
error = sum((y-round(a)).^2)
%a1 = round(sim(net_tra,x()));


