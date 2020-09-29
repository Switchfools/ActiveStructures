function [M,K,C] = moveforcematrices(masses,s,d)

    M=diag(masses);
    n=length(s);
    
    for i=1:1:n
        if i==1
            K(1,1)=s(1)+s(2);
            K(1,2)=-s(2);
        elseif i==n
            
            K(n,n)=s(n);
            K(n,n-1)=-s(n);
            
        else
            
            K(i,i)=s(i)+s(i+1);
            K(i,i-1)=-s(i);
            K(i,i+1)=-s(i+1);
            
        end
        
    end
    
    
    
    for i=1:1:n
        if i==1
            C(1,1)=d(1)+d(2);
            C(1,2)=-d(2);
        elseif i==n
            
            C(n,n)=d(n);
            C(n,n-1)=-d(n);
            
        else
            
            C(i,i)=d(i)+d(i+1);
            C(i,i-1)=-d(i);
            C(i,i+1)=-d(i+1);
            
        end
        
    end
    
    
    
    
    
    
end