clear all
close all
clc

%% lectura y adecuancion de señal de entrada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsismo='elcentro';
sismo='elcentro/elcentro_NS.mat'; %%%%%% el centro
archivo=strcat('../../inputs/',sismo);   
load(archivo)
seism=elcentro_NS*9.8; %aceleración en m/s^2 %%%%%% el centro
Tseism=time_NS(2)-time_NS(1);  %%%%%% el centro
T=Tseism;
ws=seism*1;
tseism=(0:T:Tseism*(length(ws)-1))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nsismo='kobe';
% sismo='Kobe/kobe_90.mat';     %%%%%% kobe
% archivo=strcat('../../inputs/',sismo);   
% load(archivo)
% seism=kobe_90_1*9.8; %aceleración en m/s^2  %%%%%% kobe
% Tseism=0.02;                   %%%%%% kobe
% T=Tseism;
% ws=seism*1;
% tseism=(0:T:Tseism*(length(ws)-1))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nsismo='chichi';
% sismo='chi-chi/chichi_N.mat';   %%%%%% chichi
% archivo=strcat('../../inputs/',sismo);   
% load(archivo)
% seism=chichi_N_1*9.8; %aceleración en m/s^2  %%%%%% chichi
% Tseism=0.004;                %%%%%% chichi  
% T=Tseism;
% ws=seism*1;
% tseism=(0:T:Tseism*(length(ws)-1))';


%% data, model system
masses=[3565.7 2580 2247 2057 2051 2051 2051 2051 2051]*1e3; %masees [kg]
stiffness=[919422 12913000 10431000 7928600 5743900 3292800 1674400 496420 496420]; %[N/m]
dampings=[101439 11363 10213 8904 7578 5738 4092 2228 704]; % [N s/m]
n=length(masses); % floors number

[M,K,C] = moveforcematrices(masses,stiffness,dampings);

Gamm=[1 ;zeros(n-1,1)];
Lamb=Gamm;

%% representation as a state space model¿
A=[zeros(n),eye(n); -inv(M)*K -inv(M)*C];
B=[zeros(n,1);M^(-1)*Gamm];
E=[zeros(n,1);-Lamb];

D=1;

%% numerical solution
x0=zeros(1,n*2);
tspan = [0 5]; % dominio temporal de solución
T=1e-3;
%tsol=0:T:(tspan(2));


%conversión a discreto
sys = ss(A,B,eye(2*n),zeros(2*n,1));
Gd = c2d(sys,T);
Ad=Gd.A;
Bd=Gd.B;
Cd=Gd.C;
Dd=Gd.D;



u(1)=0;%control signal
x1=x0; %define initial value
tic
for i=1:1:round(tspan(2)/T)
    
    wi(i)=interp1(tseism,ws,i*T); %interpolación datos 1D
    tsol(i)=i*T;
    
    x1(i+1,:)=Ad*x1(i,:)'+Bd*u(i)+E*wi(i)*T;
    u(i+1)=0; %uncontrolled

end
toc
tsol(i+1)=(i+1)*T;
figure
hold all
plot(tsol,x1(:,1:n))
title('positions')



% % % u(1)=0;
% % % tic
% % % for i=1:1:round(tspan(2)/T)
% % %     
% % %     %interpolation of signals
% % %     %wi(i)=interp1(t,w,i*T); %interpolación datos 1D
% % %     %tsol(i)=i*T;
% % %     
% % %     % explicit 4-stage Runge Kutta method 
% % %     [tsolc,xsolc] = ode45(@(t,x) buildingmodelOL(t,x,A,B,E,u(i),Fc,tseism,ws,T), tspan, x0);
% % %     
% % %     %LQR
% % %     
% % %     %u(i+1)=0;
% % %     
% % % 
% % % end
% % % toc
% % % figure
% % % hold all
% % % plot(tsolc,xsolc(:,1:n))
% % % title('positions')
% % % 
% % % 
% % % %T=5e-5;
% % % u(1)=0;
% % % tic
% % % x2=x0; %define initial value
% % % ta=0;
% % % for i=1:1:round(tspan(2)/T)
% % %     
% % %     %interpolation of signals
% % %     wi(i)=interp1(tseism,ws,i*T); %interpolación datos 1D
% % %     %wi(i)=0;
% % %     tsol2(i)=i*T;
% % %     
% % %     % explicit 4-stage Runge Kutta method 
% % %     k1=buildingmodel2(ta,x2(i,:)',A,B,E,u(i),wi(i));
% % %     k2=buildingmodel2(ta,x2(i,:)'+(T/2)*k1,A,B,E,u(i),wi(i));
% % %     k3=buildingmodel2(ta,x2(i,:)'+(T/2)*k2,A,B,E,u(i),wi(i));
% % %     k4=buildingmodel2(ta,x2(i,:)'+T*k3,A,B,E,u(i),wi(i));
% % %     x2(i+1,:)=(x2(i,:)'+T*((k1/6)+(k2/3)+(k3/3)+(k4/6)))';
% % %     
% % %     
% % %     u(i+1)=0;
% % %     
% % % 
% % % end
% % % toc
% % % tsol2(i+1)=(i+1)*T;
% % % figure
% % % hold all
% % % plot(tsol2,x2(:,1:n))
% % % title('positions')



filename = strcat('..\..\outputs\OL_',nsismo,'T=',num2str(T),'.mat');
save(filename)


