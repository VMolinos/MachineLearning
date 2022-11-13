% Tester
clear; close all; clc;   

%% Parámetros
fs = 8000; %Frecuencia de muestreo.
sg = 1; %Tiempo de grabación.
N = 2000; % Número de puntos de la fft, ha de ser par.


%% Diseño del banco de filtros en escala mel
% En escala mel:
ncanalesmel = 22; % Valor típico en reconocimiento automático
melmax = freq2mel(fs/2); % Máximo de la frec. Mel
melinc = melmax/(ncanalesmel+1); % Incremento en frecuencia mel
melcent = (1:ncanalesmel)*melinc; % En mel

% En frecuencia:
fcent = mel2freq(melcent); % En frecuencias
startf = [0, fcent(1:ncanalesmel-1)];
endf = [fcent(2:ncanalesmel),fs/2];

% En muestras de la fft:
fcentmues = round(fcent*N/fs); %En muestras de la fft
fstartmues = [1 , fcentmues(1:ncanalesmel-1)];
fendmues = [fcentmues(2:ncanalesmel),N/2];


% Cálculo de los coeficientes del banco de filtros:
w = zeros(ncanalesmel,N/2); % Matriz de filtros
for c = 1:ncanalesmel
    incremento = 1/(fcentmues(c)-fstartmues(c));
    for i=fstartmues(c):fcentmues(c)
        w(c,i)=(i-fstartmues(c))*incremento;
    end
    decremento = 1/(fendmues(c) - fstartmues(c));
    for i = fcentmues(c):fendmues(c)
        w(c,i) = 1 - ((i-fcentmues(c))*decremento);
    end
end

%% Grabación, procesado y almacenamiento del número.
r = 0.09; % Parámetro para el tamaño de la trama.
ltrama = floor(r*fs); % Tamaño de la trama.
D =floor(ltrama*0.8); % Desplazamiento en muestras.
silencio = 1000; % 1000 primeras y 1000 últimas muestras son silencio.
ntramas = floor((fs*sg-2*silencio-ltrama)/D)+1; % Número de tramas en total.
ncoef = 5; % Número de coeficientes que se van a tomar finalmente.

xdata = audiorecorder(fs,16,1,-1); % Inicialización para la grabación de voz.

% Grabación
repetir = 'y';
while repetir == 'y'
    fprintf('Grabación del número \n');
    
    disp('Preparado!');
    pause(2);
    disp('...3');
    pause(1);
    disp('...2');
    pause(1);
    disp('...1');
    pause(1);
    disp('ya!');
    recordblocking(xdata,sg);
    rdata = getaudiodata(xdata); %Extraccion los valores reales de la señal.
    tdata = rdata(silencio+1:length(rdata)-silencio); % Descartamos el silencio.
    disp('Reproduciendo la frase')
    soundsc(tdata,fs);
    
    figure(1)
    plot(tdata)
    repetir = input('\n¿Quieres repetir la grabación? (y/n): ','s');
    fprintf('\n\n')
    close 1
    if repetir=='y'
        clear rdata
        clear tdata
    end
end

% Procesado de la señal
% Preénfasis
for j=length(tdata):-1:2
    xp(j) = tdata(j) - tdata(j-1)*0.97;
end
xp(1) = tdata(1)*(1-0.97);
%fx = 20*log10(abs(fft(tdata))); % Espectro de la señal antes del filtrado
%fxp = 20*log10(abs(fft(xp))); % Espectro de la señal despues del filtrado

% Entramado y hamming
xt = zeros(1,ltrama);
for tr = 1:ntramas
    xt(tr,:) = xp((tr-1)*D+1:(tr-1)*D+ltrama) .* hamming(ltrama)';
end


% Cálculo de la transformada de Fourier por tramas:
fftt = zeros(ntramas,N);
for k = 1:ntramas
    fftt(k,:) = 20*log10(abs(fft(xt(k,:),N)));
end

% Salida del banco de filtros: 
fsalida = fftt(:,1:N/2) *w';

%% DCT Coeficientes cepstrales
Coef_22 = dct(fsalida);
CepsCoef(:,1) = reshape(Coef_22(:,1:ncoef),ntramas*ncoef,1);
entrada = CepsCoef(11:40,1); % Selección de la entrada.

%
load('RedEntrenada\net_tra1234.mat') % Carga de la red neuronal en el workspace.
respuesta = round(sim(net_tra1234,entrada));

% Procesado de la respuesta 1234
fprintf('Respuesta de la red: [%d %d]\n',respuesta(1),respuesta(2))
if respuesta(1)==0 && respuesta(2)==0
    disp('Has dicho el número 1')
end
if respuesta(1)==0 && respuesta(2)==1
    disp('Has dicho el número 2')
end
if respuesta(1)==1 && respuesta(2)==0
    disp('Has dicho el número 3')
end
if respuesta(1)==1 && respuesta(2)==1
    disp('Has dicho el número 4')
end

%{

load('RedEntrenada\net_tra12.mat') % Carga de la red neuronal en el workspace.
respuesta = round(sim(net_tra12,entrada));

% Procesado de la respuesta 12
fprintf('Respuesta de la red: %d\n',respuesta)
if respuesta == 0
    disp('Has dicho el número 1')
end
if respuesta == 1
    disp('Has dicho el número 2')
end
%}
