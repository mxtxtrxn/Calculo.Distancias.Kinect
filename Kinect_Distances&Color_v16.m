% 
%                           Visión Por Computadora
%                                   06.05.23
% 
% Objetivo 1: Extraer la imagen color y profundidad desde un Kinect en línea,
% y aplicar técnicas de segmentación supervisada para la localización
% (en profundidad)y segmentación de objetos color en la escena
% (al menos 3 colores).NOTA: El sistema debe identificar 7 diferentes colores,
% pero en la escena habrá máximo 3, de tal forma que,si hacemos 2 pruebas, 
% en la primera podemos poner playeras verde, azul y amarillo,y para la 
% siguiente prueba, tal vez se puede poner un naranja, rosa y azul.
% El sistema logra detectar 7 colores,aunque no estén los 7 en el escenario.
% 
% Objetivo  2: Dibujar losBounding  boxde las  personas  conlos  colores 
% detectados.  Encuentre también la localización (en profundidad)con
% respecto al Kinect y encuentre la distancia entre las personas, si
% alguiense encuentra más cerca de 1 m de distancia, enviar un cuadro 
% rojo o un mensajeque indique no sana distancia.Imprimir las distancias 
% de cada persona detectada con respecto al Kinect y de la distancia entre
% ellos.
%% Captura
clc;
close all;

% Configuraci?n del kinect para encenderlo desde matlab
%vid = videoinput('kinect',1, 'BGR_480x640');
vid = videoinput('kinect',1); %camara color
vid2 = videoinput('kinect',2); %infrarrojo
srcDepth = getselectedsource(vid);
% 
vid.FramesPerTrigger = 1;
vid2.FramesPerTrigger = 1;
% 
vid.TriggerRepeat = 1;
vid2.TriggerRepeat = 1;
% 
triggerconfig([vid vid2],'manual');
% 
start([vid vid2]); % Inicia la captura
% 
% % Trigger 200 times to get the frames.
for i = 1:1
%     % Trigger both objects.
     trigger([vid vid2])
%     % Get the acquired frames and metadata.
     [A, ts_color, metaData_Color] = getdata(vid);
     [imgdisp, ts_depth, metaData_Depth] = getdata(vid2);
end
 
% mostrar imagen de profundidad
 
figure(1)
imshow(imgdisp, [min(min(imgdisp)) max(max(imgdisp))]);
dap=double(imgdisp)/double(max(max(imgdisp))).*255; % normaliza y expande el valor de 0 a 255
%dap=double(imgdisp)/double(max(max(imgdisp))); % normaliza el valor de 0 a 1
imwrite(uint8(dap),'C:\Users\doral\Documents\img2_disp.png', 'png'); % Esta instruccion guarda la imagen de profundidad, con el nombre img_4.png
%   
imwrite(imgdisp,'C:\Users\doral\Documents\img1_disp_double.tif');% en la ruta donde tengas este archivo
% A = imread('img_1m.png');
figure(2)
imshow(A)  % imshow muestra la imagen color
imwrite(A, 'C:\Users\doral\Documents\img1_color.png', 'png');  % Esta instruccion guarda la imagen a color, con el nombre img_4.png
%                                                               en la ruta donde tengas este archivo
stop([vid vid2]); % Termina la captura

%% Parte 2: Programa 
clc;
close all;

im = imread('img1_color.png');
imG = imread('img1_disp.png');
imCIE = rgb2lab(im);
[m,n,ch] = size(imCIE);
imRes_aux = zeros(m,n,ch);
imBin = zeros(m,n);
wi=640; 
he=480;
x=0; 
y=40; 
dx=30; 
dy=0; 
dwi=60;
dhe=180; 

%recortar el área de interés
imR=imcrop(im,[(x+dx) (y-dy) (wi-dwi) (he-dhe)]); 
imG=imcrop(imG, [(x+dx) (y-dy) (wi-dwi) (he-dhe)]); 
np = 2;
figure(1)
imshow(imR)
title('Imagen RGB')
figure(2)
imshow(imG)
title('Imagen profundidad')

[u,v, ch] = size(imR);
[uG, vG, chG]=size(imG); 
level= multithresh(imG,np);
seg_I = imquantize(imG, level);

%                   D I C C I O N A R I O ( L A B )     
L= [43.16,33.30,40.65,...       %1.- morado claro
    73.06,66.84,49.61,...       %2.- amarillo
    15.60,18.63,16.82,...       %3.- vino
    36.98,36.50,38.96,...       %4.- verde
    14.39,11.84,14.10,...       %5.- azul marino
    26.89,29.52,22.38,...       %6.- azul
    13.43,12.11,11.23,...       %7.- morado oscuro
    7.80,5.22,6.90];            %8.- cafe 

A= [17.09, 27.22,13.22,...
    10.26,-0.42,8.36,...
    24.86,24.81,24.21,...
    -11.05,-6.86,-9.34,...
    4.76,4.56,7.65,...
    -13.45,-1.50,8.57,...
    17.62,12.52,13.86,...
    10.33,6.25,10.40];

B= [-33.206,-49.10,-30.03,...
    35.31,65.93,51.01,...
    0.49,-0.63,5.95,...
    2.92,2.92,4.01,...
    -11.87,-13.02,-13.01,...
    6.04,-22.36,-31.14,...
    -18.45,-11.76,-17.52,...
    -2.00,-2.91,-2.60];

for k=1:24
%valor del color a segmentar
Lab_ref = [L(k);A(k);B(k)]; 
imRes_aux(:,:,1) = (imCIE(:,:,1)-Lab_ref(1)).^2;
imRes_aux(:,:,2) = (imCIE(:,:,2)-Lab_ref(2)).^2;
imRes_aux(:,:,3) = (imCIE(:,:,3)-Lab_ref(3)).^2;
imR_aux = (imRes_aux(:,:,1)+imRes_aux(:,:,2)+imRes_aux(:,:,3)).^(1/2);
%normalizando la imagen resta
imR_aux= imR_aux/max(max(imR_aux));
%nivel de umbral
th_aux=0.12;
imBin(imR_aux<th_aux)=1;
imRes_aux(:,:,1)=single(im(:,:,1)).*single(imBin(:,:));
imRes_aux(:,:,2)=single(im(:,:,2)).*single(imBin(:,:));
imRes_aux(:,:,3)=single(im(:,:,3)).*single(imBin(:,:));
figure (2)
imshow(uint8(imRes_aux))
end

imB= imbinarize(imG, 0.2);

% figure(4)
% imhist(imG);

p_grises = bwareaopen(imB,100); % elimina las pequeñas manchas blancas
a_grises=regionprops(p_grises);

imRes_aux=imcrop(imRes_aux,[(x+dx) (y-dy) (wi-dwi) (he-dhe)]); 

figure(3); 
imshow(uint8(imRes_aux)); 
title('Figuras a analizar'); 
hold on; 

plot(ceil(a_grises(1).Centroid(1)),ceil(a_grises(1).Centroid(2)), 'b*'); 
hold on; 
plot(ceil(a_grises(2).Centroid(1)-10),ceil(a_grises(2).Centroid(2)-10), 'b*');
hold on; 
plot(ceil(a_grises(3).Centroid(1)-10),ceil(a_grises(3).Centroid(2)-10), 'b*'); 
hold on;

x(1)=a_grises(1).Centroid(1); 
x(2)=a_grises(2).Centroid(1); 
x(3)=a_grises(3).Centroid(1); 
y(1)=a_grises(1).Centroid(2); 
y(2)=a_grises(2).Centroid(2); 
y(3)=a_grises(3).Centroid(2);

x=round (x);
y=round (y);

ng = 255;
imn=imR/ng;
imLab=rgb2lab(imn); 

auxIma = zeros(3,u*v);
imren=imLab(:,:,1); 
auxIma(1,:)=imren(:); 
imren=imLab(:,:,2); 
auxIma(2,:)=imren(:);
imren=imLab(:,:,3); 
auxIma(3,:)=imren(:);
imRes= zeros(u,v,np,3); 

for k=1:np
imRef=[imLab(y(k), x(k), 1); imLab(y(k), x(k), 2) ; imLab(y(k), x(k), 3)];
imSeg = sqrt((auxIma(1,:)-imRef(1)).^2 +(auxIma(2, :)-imRef(2)).^2 + (auxIma(3, : )-imRef(3)).^2);

imProb = zeros (u, v) ;
imProb(:)=imSeg/max(imSeg);
imB = zeros(u,v);
imB (imProb < th_aux) = 1;

se = strel( 'line', 3, 0); % Limpiar mascara
imFilt = imerode(imG, se) ;

imRes(:,:,1,k)= single(imFilt).*single(imR(:,:,1));
imRes(:,:,2,k)= single(imFilt).*single(imR(:,:,2));
imRes(:,:,3,k)= single(imFilt).*single(imR(:,:,3));
end
 for i=2:np
imRes(:,:,:,1) = imRes(:,:,:,1) + imRes(:,:,:,i);
 end
mask_color=rgb2gray(imRes(:,:,:,1));
p = bwareaopen(mask_color, 300);
a_color=regionprops(p);

figure(4);
imshow(uint8(imRes_aux)); 
title('objetos segmentados'); 
hold on; 

rectangle('Position', a_color(1).BoundingBox, 'EdgeColor','b', 'LineWidth', 1.5); 
plot(ceil(a_color(1).Centroid(1)),ceil(a_color(1).Centroid(2)), 'b*'); 
hold on;
rectangle('Position', a_color(2).BoundingBox, 'EdgeColor','b', 'LineWidth', 1.5); 
plot(ceil(a_color(2).Centroid(1)-10),ceil(a_color(2).Centroid(2)-10), 'b*'); 
hold on; 
rectangle('Position', a_color(3).BoundingBox, 'EdgeColor','b', 'LineWidth', 1.5); 
plot(ceil(a_color(3).Centroid(1)-10),ceil(a_color(3).Centroid(2)-10), 'b*'); 
hold on;


%                           Distancias en X
%                       Calculo de resolucion
n=3;
[pxFotoX, pxFotoY, Canales] = size(imRes_aux);
mObjCercano = 0.6;

%                           pendiente
pxObjCercanoX(1) = a_color(1).BoundingBox(3);
pxObjCercanoX(2) = a_color(1).BoundingBox(3);
pxObjCercanoX(3) = a_color(1).BoundingBox(3);
resolucion(1) = mObjCercano / pxObjCercanoX(1);
resolucion(2) = mObjCercano / pxObjCercanoX(2);
resolucion(3) = mObjCercano / pxObjCercanoX(3);

%           Calculo de distancias entre objetos en pixeles
distanciaPX = zeros(n);
distanciaPX(1) = a_color(2).Centroid(1) - a_color(1).Centroid(1);
distanciaPX(2) = a_color(3).Centroid(1) - a_color(2).Centroid(1);
distanciaPX(3) = a_color(3).Centroid(1) - a_color(1).Centroid(1);

%                   Conversion de distancia a metros
distanciaMX(1) = distanciaPX(1) * resolucion(1);
distanciaMX(2) = distanciaPX(2) * resolucion(2);
distanciaMX(3) = distanciaPX(3) * resolucion(3);

%                           Distancias en Z
aux = impixel(imG, a_grises(1).Centroid(1), a_grises(1).Centroid(2));
distanciaZ(1) = 0.015 * aux(1) + 0.05;

aux = impixel(imG, a_grises(2).Centroid(1), a_grises(2).Centroid(2));
distanciaZ(2) = 0.015 * aux(1) + 0.05;

aux = impixel(imG, a_grises(3).Centroid(1), a_grises(3).Centroid(2));
distanciaZ(3) = 0.015 * aux(1) + 0.05;

distanciaMZ(1) = distanciaZ(2) - distanciaZ(1);
distanciaMZ(2) = distanciaZ(3) - distanciaZ(2);
distanciaMZ(3) = distanciaZ(3) - distanciaZ(1);

distanciaTotal(1) = sqrt(distanciaMX(1) * distanciaMX(1) + distanciaMZ(1) * distanciaMZ(1));
distanciaTotal(2) = sqrt(distanciaMX(2) * distanciaMX(2) + distanciaMZ(2) * distanciaMZ(2));
distanciaTotal(3) = sqrt(distanciaMX(3) * distanciaMX(3) + distanciaMZ(3) * distanciaMZ(3));

if distanciaTotal(1) < 1.5
    figure(5);
    imshow(uint8(imRes_aux)); 
    title('Peligro:No se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(1).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(1).Centroid(1)),ceil(a_color(1).Centroid(2)), 'r*'); 
    hold on; 
    rectangle('Position', a_color(2).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(2).Centroid(1)),ceil(a_color(2).Centroid(2)), 'r*'); 
    hold on;
else 
    figure(5);
    imshow(uint8(imRes_aux)); 
    title('Se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(1).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(1).Centroid(1)),ceil(a_color(1).Centroid(2)), 'g*'); 
    hold on; 
    rectangle('Position', a_color(2).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(2).Centroid(1)),ceil(a_color(2).Centroid(2)), 'g*');
end

if distanciaTotal(2) < 1.5
    figure(6);
    imshow(uint8(imRes_aux)); 
    title('Peligro:No se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(2).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(2).Centroid(1)),ceil(a_color(2).Centroid(2)), 'r*'); 
    hold on; 
    rectangle('Position', a_color(3).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(3).Centroid(1)),ceil(a_color(3).Centroid(2)), 'r*'); 
else 
    figure(6);
    imshow(uint8(imRes_aux)); 
    title('Se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(3).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(3).Centroid(1)),ceil(a_color(3).Centroid(2)), 'g*'); 
    hold on; 
    rectangle('Position', a_color(2).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(2).Centroid(1)),ceil(a_color(2).Centroid(2)), 'g*');
end

if distanciaTotal(3) < 1.5
    figure(7);
    imshow(uint8(imRes_aux)); 
    title('Peligro:No se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(1).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(1).Centroid(1)),ceil(a_color(1).Centroid(2)), 'r*'); 
    hold on; 
    rectangle('Position', a_color(3).BoundingBox, 'EdgeColor','r', 'LineWidth', 1.5); 
    plot(ceil(a_color(3).Centroid(1)),ceil(a_color(3).Centroid(2)), 'r*'); 
else 
    figure(7);
    imshow(uint8(imRes_aux)); 
    title('Se cumple sana distancia'); 
    hold on; 
    rectangle('Position', a_color(1).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(1).Centroid(1)),ceil(a_color(1).Centroid(2)), 'g*'); 
    hold on; 
    rectangle('Position', a_color(3).BoundingBox, 'EdgeColor','g', 'LineWidth', 1.5); 
    plot(ceil(a_color(3).Centroid(1)),ceil(a_color(3).Centroid(2)), 'g*');
end

fprintf("Distancia total entre elemento 1 y 2: %0.3f m \n\n",uint8(distanciaTotal(1)));
fprintf("Distancia total entre elemento 2 y 3: %0.3f m \n\n",uint8(distanciaTotal(2)));
fprintf("Distancia total entre elemento 1 y 3: %0.3f m \n\n",uint8(distanciaTotal(3)));

