function dxdt = buildingmodel2(t,x,A,B,E,u,wi)
    dxdt = zeros(18,1);
    
    
    dxdt=A*x+B*u+E*wi;


end