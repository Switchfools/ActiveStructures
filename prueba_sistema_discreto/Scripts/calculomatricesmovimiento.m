clear all;
close all;
clc

%% lectura y adecuancion de señal de entrada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsismo='elcentro';
sismo='elcentro/elcentro_NS.mat'; %%%%%% el centro
archivo=strcat('../../inputs/',sismo);   
load(archivo)
seism=elcentro_NS*9.8*1.05; %aceleración en m/s^2 %%%%%% el centro
Tseism=time_NS(2)-time_NS(1);  %%%%%% el centro
T=Tseism;
ws=seism*1;
tseism=(0:T:Tseism*(length(ws)-1))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nsismo='kobe';
% sismo='Kobe/kobe_90.mat';     %%%%%% kobe
% archivo=strcat('../../inputs/',sismo);   
% load(archivo)
% seism=kobe_90_1*9.8*1.05; %aceleración en m/s^2  %%%%%% kobe
% Tseism=0.02;                   %%%%%% kobe
% T=Tseism;
% ws=seism*1;
% tseism=(0:T:Tseism*(length(ws)-1))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nsismo='chichi';
% sismo='chi-chi/chichi_N.mat';   %%%%%% chichi
% archivo=strcat('../../inputs/',sismo);   
% load(archivo)
% seism=chichi_N_1*9.8*1.05; %aceleración en m/s^2  %%%%%% chichi
% Tseism=0.004;                %%%%%% chichi  
% T=Tseism;
% ws=seism*1;
% tseism=(0:T:Tseism*(length(ws)-1))';



%%  Structural Characteristic definition
% Model parameters of the 20-story benchmark building structure
m(1)= 1.126; % Mass (10^6) Kg
m(2:19)=1.1;
m(20)= 1.17;
m=m*1e6; %kg

k(1:5) = 862.07; % Interstory stiffness (10^3) KN/m;
k(6:11) = 554.17;
k(12:14) = 453.51;
k(15:17) = 291.23;
k(18:19) = 256.46;
k(20)=171.70;
k=k*1e3*1e3; % N/m

% Equation of motions
% Dynamics of the structure. System Matrices
M = diag(m); % Mass Matrix

K=zeros(length(k)); % Stiffness Matrix
for i=1:length(K)
    for j=1:length(K)
        if j==length(K) && i ==length(K)
            K(j,j)= k(end);
            
        elseif i == j
            K(i,i)=k(i)+k(i+1);
            
        end
    end
end

for i=1:length(K)-1
    K(i,i+1)=-k(i+1);
end

for j=1:length(K)-1
    K(j+1,j)=-k(j+1);
end

%Procedure to determine damping matrix C
w=sqrt(eig(M\K));

N=20;
R=chol(M);
L=R';
AA=inv(L)*K*inv(L');
[x,WW]=eig(AA);
v=inv(L')*x;

for i =1:N
    w1(i)=sqrt(WW(i,i));
end

[w,I]=sort(w1);

% Let us determine damping ratio using Rayleigh
% Using 5% damping on the first 2 modes(Lei et al), solve for a0 and a1
% Frist 10 natural frequencies (Ohtori et al) are 0.261, 0.753, 1.30,
% 1.83, 2.4, 2.44, 2.92, 3.01, 3.63 and 3.68 Hz
% Wn =[0.261, 0.753, 1.30,1.83, 2.4, 2.44, 2.92, 3.01, 3.63, 3.68];
% %Spencer 1998 actual
% Wn = [0.28 0.74 1.23 1.66 2.09]; %Lynch Thesis in Hz (multiply by 2*pi to get rad/sec)
Wn=sort(sqrt(eig(M\K)));
w1= Wn(1);
w2= Wn(2);

W=[1/w1 w1; 1/w2 w2];

zeta = 2*[0.05;0.05];

syms aa0 aa1
[aa0,aa1]=solve(W*[aa0;aa1]==zeta);

aa0=sym2poly(aa0);
aa1=sym2poly(aa1);

C = aa0*M + aa1*K; % Damping Matrix

%%

p=20; %pisos
xk=zeros(2*p,1); % definicion del valor actual como las condiciones inicales
% coeficiente de control Tu ones in the diagonal, -1 diagonal sup
Tu=eye(p);
for i=1:1:p-1 % put -1 in the upper diagonal
    Tu(i,i+1)=-1;
end

A=[zeros(p),eye(p);-inv(M)*K,-inv(M)*C];
B=[zeros(p);inv(M)*Tu];
E=[zeros(p,1);-ones(p,1)];



