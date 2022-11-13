%% Red Neuronal
close all;clear;clc;

%% Entradas
ncoef = 3; % N�mero de coeficientes considerados.
limite = 10*(ncoef+1);% L�mite para tomar ncoef coeficientes.
ngrabaciones = 100; % N�mero de grabaciones consideradas.

% Datos de V�ctor 
xdigV1 = load('Coeficientes/Uno/uno.mat');
xdigV1 = xdigV1.CepsCoef(11:limite,1:ngrabaciones);
xdigV2 = load('Coeficientes/Dos/dos.mat');
xdigV2 = xdigV2.CepsCoef(11:limite,1:ngrabaciones);
xdigV3 = load('Coeficientes/Tres/tres.mat');
xdigV3 = xdigV3.CepsCoef(11:limite,1:ngrabaciones);
xdigV4 = load('Coeficientes/Cuatro/cuatro.mat');
xdigV4 = xdigV4.CepsCoef(11:limite,1:ngrabaciones);

% Matriz de entrada
x = horzcat(xdigV1,xdigV2,xdigV3,xdigV4);
[m,n] = size(x);

% Representaci�n de coeficientes
idx = 1;
figure(idx)
mesh(x); title('Coeficientes')
xlabel('grabaci�n');ylabel('muestras/coeficientes');zlabel('Valor');


%% Matriz de salidas

y1 = zeros(1,n);
y1(1,201:400) = ones(1,200); 

y2 = zeros(1,n);
y2(1,101:200) = ones(1,100);
y2(1,301:400) = ones(1,100);

y = vertcat(y1,y2);

idx = 1;
figure(idx)
subplot(2,1,1)
stem(y1)
subplot(2,1,2)
stem(y2)

%% Red
S = [30,12,5]; % Topolog�a de la red. Cantidad de neuronas en cada layer.
TF = {'logsig','logsig','purelin'}; % Funci�n de activaci�n en cada layer

net = newff(x,y,S); % Configuraci�n de la red (Feed Forward)
net.trainFcn = 'traingd'; % Algoritmo de entrenamiento (Backpropagation, descendent gradient)
net.trainParam.epochs=1000000; % Cantidad m�xima de iteracciones.


[net_tra1234,~,a,~] = train(net,x,y);
%save('RedEntrenada/net_tra1234','net_tra1234') % Almacenamiento de la red entrenada
a = round(a); % �ltima capa 
error = sum(sum((y-round(a)).^2)); % Error
fprintf('El error es: %d \n',error);
