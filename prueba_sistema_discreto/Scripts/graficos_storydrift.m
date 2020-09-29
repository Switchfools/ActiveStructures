close all
clear all
clc


pos=1;
sismon='elcentro';
%sismon='kobe';
%sismon='chichi';


h1=figure(1);
axes1 = axes('Parent',h1,'FontSize',15);
set(gcf, 'Position', get(0,'Screensize'))% ampliar la gráfica a toda la pantalla
%set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 50 30])
xlim(axes1,'auto'); % definicion de los limites del eje x 
ylim(axes1,'auto'); % definicion de los limites del eje y 
box(axes1,'on');% grafica el borde de los ejes al rededor de la gráfia
hold(axes1,'all'); %acotar los ejes a los limites definidos(aplicar axes1)


load(strcat('../outputs/OL_',sismon,'T=0.001.mat'))
for i=1:1:8
    drift(:,i)=abs(x1(:,i)-x1(:,i+1));
end
msd=max(drift); %max story drift
%plot(msd,1:1:8)
semilogx(msd,1:1:8,'*-b','linewidth',2)


% % % clear drift
% % % load(strcat('../outputs/LQR_',sismon,'T=5e-05.mat'))
% % % for i=1:1:8
% % %     drift(:,i)=abs(x(:,i)-x(:,i+1));
% % % end
% % % msd=max(drift); %max story drift
% % % semilogx(msd,1:1:8,'*-c','linewidth',2)
% % % clear drift


xlabel('Displacement [m]','Interpreter','latex','FontSize',22);
ylabel('Story Drift','Interpreter','latex','FontSize',22);
legend('Noncontrolled')

%set(gca, 'XScale', 'log')