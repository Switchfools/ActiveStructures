function dxdt = buildingmodelOL(t,x,A,B,E,u,Fc,ts,w,T)
    dxdt = zeros(18,1);
    
    wi=interp1(ts,w,t); %interpolación datos 1D
    %wi=0;
    
    u=0;
    dxdt=A*x+B*u+E*wi;





end