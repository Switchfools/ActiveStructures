import numpy as np
import control as con
import matplotlib.pyplot as plt
def GenerateMatrices(Mass,Stiffness,Damping):
    M=np.diag(Mass)
    shapesS=np.shape(Stiffness)
    K=np.zeros((shapesS[0],shapesS[0]))
    G=np.zeros((shapesS[0],shapesS[0]))
    for i in range(shapesS[0]):
        if(i==0):
            K[i,i]=Stiffness[i]+Stiffness[i+1]
            K[i,i+1]=-Stiffness[i+1]
            G[i,i]=Damping[i]+Damping[i+1]
            G[i,i+1]=-Damping[i+1]
        elif(i==shapesS[0]-1):
            K[i,i]=Stiffness[i]
            K[i,i-1]=-Stiffness[i]
            G[i,i]=Damping[i]
            G[i,i-1]=-Damping[i]
        else:
            K[i,i]=Stiffness[i]+Stiffness[i+1]
            K[i,i+1]=-Stiffness[i+1]
            K[i,i-1]=-Stiffness[i]
            G[i,i]=Damping[i]+Damping[i+1]
            G[i,i+1]=-Damping[i+1]
            G[i,i-1]=-Damping[i]
    T_u=np.zeros((shapesS[0],1))
    T_u[0]=1
    T_w=T_u
    return(M,K,G,T_u,T_w)

def ToStateSpace(M,K,G,T_u,T_w):
    shape=np.shape(M)
    A=np.block([[np.zeros((shape[0],shape[0])), np.identity(shape[0])], [-np.inv(M)*K, -np.inv(M)*G]])
    B=np.block([[np.zeros((shape[0],1)), np.identity(shape[0])], [-np.inv(M)*K, -np.inv(M)*G]])
    return(sys)

# System parameters
m = 4       # mass of aircraft
J = 0.0475  # inertia around pitch axis
r = 0.25    # distance to center of force
g = 9.8     # gravitational constant
c = 0.05    # damping factor (estimated)

# State space dynamics
xe = [0, 0, 0, 0, 0, 0]  # equilibrium point of interest
ue = [0, m*g]  # (note these are lists, not matrices)
A = np.matrix(
    [[0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, (-ue[0]*np.sin(xe[2]) - ue[1]*np.cos(xe[2]))/m, -c/m, 0, 0],
     [0, 0, (ue[0]*np.cos(xe[2]) - ue[1]*np.sin(xe[2]))/m, 0, -c/m, 0],
     [0, 0, 0, 0, 0, 0]]
)

# Input matrix
B = np.matrix(
    [[0, 0], [0, 0], [0, 0],
     [np.cos(xe[2])/m, -np.sin(xe[2])/m],
     [np.sin(xe[2])/m, np.cos(xe[2])/m],
     [r/J, 0]]
)

# Output matrix
C = np.matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
D = np.matrix([[0, 0], [0, 0]])

sys = con.StateSpace(A, B, C, D)
sysd = sys.sample(0.5, method='bilinear')

# Label the plot
plt.clf()
plt.suptitle("LQR controllers for vectored thrust aircraft (pvtol-lqr)")
plt.show()
