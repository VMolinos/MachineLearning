clear; close all; clc;   

%% Par�metros
fs = 8000; %Frecuencia de muestreo.
sg = 1; %Tiempo de grabaci�n.
N = 2000; % N�mero de puntos de la fft, ha de ser par


%% Elecci�n del modo
fprintf('Elije un modo \n1: Grabar y obtener coeficientes  \n2: Reproducir una grabaci�n en concreto y ver los coeficientes.\nModo: ')
modo = input('');
fprintf('\n')

while (modo<1)||(modo>2)
    modo = input('Elije solamente 1 � 2: '); 
    fprintf('\n')
end

%% Modos de funcionamiento
switch modo
    case 1
        %% Dise�o del banco de filtros en escala mel
        % En escala mel:
        ncanalesmel = 22; % Valor t�pico en reconocimiento autom�tico
        melmax = freq2mel(fs/2); % M�ximo de la frec. Mel
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
        
        
        % C�lculo de los coeficientes del banco de filtros:
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

        % Introducci�n del n�mero deseado y la cantidad de grabaciones
        % consecutivas
        NUMERO = input('Introduce el n�mero a grabar: ');
        niteracciones = input('Introduce el n�mero de veces que quieres grabar el n�mero: ');
        fprintf('\n')
        while (0>NUMERO)||(NUMERO>9)
            NUMERO = input('Debes introducir un n�mero entre 0 y 9: ');
            fprintf('\n')
        end

        switch NUMERO
            case 0
                num = 'Coeficientes/Cero/cero';
                au = 'Grabaciones/Cero/cero';
                auxnum = 'CERO';
            case 1
                num = 'Coeficientes/Uno/uno';
                au = 'Grabaciones/Uno/uno';
                auxnum = 'UNO';
            case 2
                num = 'Coeficientes/Dos/dos';
                au = 'Grabaciones/Dos/dos';
                auxnum = 'DOS';
            case 3
                num = 'Coeficientes/Tres/tres';
                au = 'Grabaciones/Tres/tres';
                auxnum = 'TRES';
            case 4
                num = 'Coeficientes/Cuatro/cuatro';
                au = 'Grabaciones/Cuatro/cuatro';
                auxnum = 'CUATRO';
            case 5
                num = 'Coeficientes/Cinco/cinco';
                au = 'Grabaciones/Cinco/cinco';
                auxnum = 'CINCO';
            case 6
                num = 'Coeficientes/Seis/seis';
                au = 'Grabaciones/Seis/seis';
                auxnum = 'SEIS';
            case 7
                num = 'Coeficientes/Siete/siete';
                au = 'Grabaciones/Siete/siete';
                 auxnum = 'SIETE';
            case 8
                num = 'Coeficientes/Ocho/ocho';
                au = 'Grabaciones/Ocho/ocho';
                auxnum = 'OCHO';
            case 9
                num = 'Coeficientes/Nueve/nueve';
                au = 'Grabaciones/Nueve/nueve';
                auxnum = 'NUEVE';
        end
        
        
        
        %% Grabaci�n, procesado y almacenamiento del n�mero.
        r = 0.09; % Par�metro para el tama�o de la trama.
        ltrama = floor(r*fs); % Tama�o de la trama.
        D =floor(ltrama*0.8); % Desplazamiento en muestras.
        silencio = 1000; % 1000 primeras y 1000 �ltimas muestras son silencio.
        ntramas = floor((fs*sg-2*silencio-ltrama)/D)+1; % N�mero de tramas en total.
        ncoef = 5; % N�mero de coeficientes que se van a tomar finalmente.
        
        % Inicializaciones
        CepsCoef = zeros(ncoef*ntramas,niteracciones);
        audio = zeros(fs*sg-2*silencio,niteracciones);
        xdata = audiorecorder(fs,16,1,-1);
        
        
        % Grabaci�n
        for i=1:niteracciones
            repetir ='y';
            while repetir=='y'
    
                fprintf('Grabaci�n %d del n�mero: %s \n',i,auxnum);
            
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
                rdata = getaudiodata(xdata); %Extraccion los valores reales de la se�al.
                tdata = rdata(silencio+1:length(rdata)-silencio); % Descartamos las primeras 1000 muestras de silencio.
                audio(:,i) = tdata;
    
                % Procesado de la se�al
    
                % Pre�nfasis
                for j=length(tdata):-1:2
                    xp(j) = tdata(j) - tdata(j-1)*0.97;
                end
                xp(1) = tdata(1)*(1-0.97);
                %fx = 20*log10(abs(fft(tdata))); % Espectro de la se�al antes del filtrado
                %fxp = 20*log10(abs(fft(xp))); % Espectro de la se�al despues del filtrado
    
                % Entramado y hamming
                xt = zeros(1,ltrama);
                for tr = 1:ntramas
                    xt(tr,:) = xp((tr-1)*D+1:(tr-1)*D+ltrama) .* hamming(ltrama)';
                end
    
    
                % C�lculo de la transformada de Fourier por tramas:
                fftt = zeros(ntramas,N);
                for k = 1:ntramas
                    fftt(k,:) = 20*log10(abs(fft(xt(k,:),N)));
                end
    
                % Salida del banco de filtros: 
                fsalida = fftt(:,1:N/2) *w';
    
                % DCT Coeficientes cepstrales
                Coef_22 = dct(fsalida);
                CepsCoef(:,i) = reshape(Coef_22(:,1:ncoef),ntramas*ncoef,1);
    
                figure(1)
                plot(tdata)
                disp('Reproduciendo la frase')
                soundsc(tdata,fs);
       
                repetir = input('\n�Quieres repetir la grabaci�n? (y/n): ','s');

                fprintf('\n\n')
                close 1
            end
        end
        
        % Almacenamiento del audio y los coeficientes
        save(num,'CepsCoef')
        save(au,'audio');

    case 2
        NUMERO = input('Indica que n�mero quieres inspeccionar: ');

        while (0>NUMERO)||(NUMERO>9)
            NUMERO = input('Debes introducir un n�mero entre 0 y 9: ');
        end
       
        switch NUMERO
            case 0
                num = 'Coeficientes/Cero/cero.mat';
                au = 'Grabaciones/Cero/cero.mat';
            case 1
                num = 'Coeficientes/Uno/uno.mat';
                au = 'Grabaciones/Uno/uno.mat';
            case 2
                num = 'Coeficientes/Dos/dos.mat';
                au = 'Grabaciones/Dos/dos.mat';
            case 3
                num = 'Coeficientes/Tres/tres.mat';
                au = 'Grabaciones/Tres/tres.mat';
            case 4
                num = 'Coeficientes/Cuatro/cuatro.mat';
                au = 'Grabaciones/Cuatro/cuatro.mat';
            case 5
                num = 'Coeficientes/Cinco/cinco.mat';
                au = 'Grabaciones/Cinco/cinco.mat';
            case 6
                num = 'Coeficientes/Seis/seis.mat';
                au = 'Grabaciones/Seis/seis.mat';
            case 7
                num = 'Coeficientes/Siete/siete.mat';
                au = 'Grabaciones/Siete/siete.mat';
            case 8
                num = 'Coeficientes/Ocho/ocho.mat';
                au = 'Grabaciones/Ocho/ocho.mat';
            case 9
                num = 'Coeficientes/Nueve/nueve.mat';
                au = 'Grabaciones/Nueve/nueve.mat';
        end
        
        % Audio
        audio = load(au);
        audio = audio.audio;
        [na,ma] = size(audio);
        
        grabacion = input('Indica el n�mero de la grabacion que quieres escuchar: ');
        fprintf('\n')

        while (1>grabacion)||(grabacion>ma)
            fprintf('Debes elegir un valor entre 1 y %d: ',ma);
            grabacion = input('');
            fprintf('\n')

        end
        
        disp('Reproduciendo la frase')
        soundsc(audio(:,grabacion),fs);

        
        % Coeficientes
        load(num);
        coef = CepsCoef(:,grabacion);
        figure()
        plot(coef);
end
        



