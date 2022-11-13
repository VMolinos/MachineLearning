%% Red Neuronal
close all;clear;clc;

%% Entradas
ncoef = 3; % Número de coeficientes considerados.
limite = 10*(ncoef+1);% Límite para tomar ncoef coeficientes.
ngrabaciones = 100; % Número de grabaciones consideradas.

% Datos de Víctor 
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

% Representación de coeficientes
idx = 1;
figure(idx)
mesh(x); title('Coeficientes')
xlabel('grabación');ylabel('muestras/coeficientes');zlabel('Valor');


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
S = [30,12,5]; % Topología de la red. Cantidad de neuronas en cada layer.
TF = {'logsig','logsig','purelin'}; % Función de activación en cada layer

net = newff(x,y,S); % Configuración de la red (Feed Forward)
net.trainFcn = 'traingd'; % Algoritmo de entrenamiento (Backpropagation, descendent gradient)
net.trainParam.epochs=1000000; % Cantidad máxima de iteracciones.


[net_tra1234,~,a,~] = train(net,x,y);
%save('RedEntrenada/net_tra1234','net_tra1234') % Almacenamiento de la red entrenada
a = round(a); % Última capa 
error = sum(sum((y-round(a)).^2)); % Error
fprintf('El error es: %d \n',error);
