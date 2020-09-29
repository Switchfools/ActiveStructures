import numpy as np
import pandas as pd
import control as con
import matplotlib.pyplot as plt
from numba import jit
import matplotlib.animation as animation
import scipy.io as sio
import gym

def LoadSeismicData(Earthquake, extension):
    path="prueba_sistema_discreto/inputs/"+Earthquake+"/"+Earthquake+extension
    mat_contents = sio.loadmat(path)
    return mat_contents["elcentro_NS"],mat_contents["time_NS"]

def SaveModel(ModelName):
    masses=np.array([3565.7, 2580 ,2247, 2057, 2051 ,2051 ,2051 ,2051 ,2051])*1e3 #mass[kg]
    stiffness=np.array([919422, 12913000, 10431000, 7928600, 5743900, 3292800, 1674400, 496420, 496420])# [N/m]
    dampings=np.array([101439, 11363, 10213, 8904, 7578, 5738, 4092, 2228, 704]) # [N s/m]
    np.savez('prueba_sistema_discreto/Models/'+ModelName+'.npz', Masses=masses, Stiffness=stiffness,Dampings=dampings)

def LoadDatamodel(Model):
    path="prueba_sistema_discreto/Models/"+Model+".npz"
    Modelito=np.load(Model+".npz")
    Mass=Modelito["Masses"]
    Stiffness=Modelito["Stiffness"]
    Damping=Modelito["Dampings"]
    return(Mass,Stiffness,Damping)

def GenerateMatricess(Mass,Stiffness,Damping):
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
def GenerateModel(Mass,Stiffness,Damping):
    n=len(Mass)
    M,K,G,T_u,T_w=GenerateMatricess(Mass,Stiffness,Damping)
    A=np.block([[np.zeros((n,n)), np.identity(n)],[-np.linalg.inv(M)@K, -np.linalg.inv(M)@G]])
    B_s=np.block([[np.zeros((n,1))], [-np.linalg.inv(M)@T_u]])
    print(np)
    E=np.block([[np.zeros((n,1))], [-T_w]])
    B=np.block([B_s, E])
    D=[1,0]
    C=np.block([[np.zeros((1,n)),np.ones((1,n))]])
    sys = con.StateSpace(A, B, C, D)
    return sys

def SimStep(sys,x_0,U_j):
    print("step")
    return np.zeros((np.shape(sys.A)[0], 1))
def WholeSim(sys,U_t,T):
    x_0=np.zeros((np.shape(sys.A)[0], 1))
    x=np.zeros((np.shape(sys.A)[0], len(T)))
    y=np.zeros((np.shape(sys.A)[0], len(T)))
    for j in range(len(T)):
        U_j=U_t[:,j]
        x_n=SimStep(sys,x_0,U_j)
        x[:,j]=x_n+x_0
    return x,y,t

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def init():
    """initialize animation"""
    Floors.set_data([], [])
    return Floors
def animate(i):
    """perform animation step"""
    global m
    n=np.shape(x)
    if(i%5==0):
        m+=1
    #print(m)
    # update pieces of the animation
    Floors.set_data(x[len(M):,m%n[1]], range(int(n[0]/2)))
    return Floors
if __name__ == "__main__":
    #We define the seismic activity
    global m
    m=0
    ##Choose Any earthquake from the list
    EQ=["elcentro","chi-chi","Kobe"]
    Ext=["_NS.mat","_NS.mat","_NS.mat"]
    W_t,t=LoadSeismicData(EQ[0], Ext[0])
    U_t=np.block([[np.zeros((1,len(t)))],[W_t.T*9.8]])
    print(U_t)
    #we define the building paramters
    ##Choose Any model of the list
    models=["NineFloors", "ThreeFloors","FiveFloors"]
    M,S,D=LoadDatamodel(models[1])
    Model=GenerateModel(M,S,D)
    x,y,t=WholeSim(Model,U_t,t.T)
    #We simulate the response
    t, y, x = con.forced_response(Model, U=U_t, T=t.T)
    for j in range(len(M)):
        plt.plot(t[:300], x[len(M)+j,:300], label="Floor: "+str(j))
    #plt.plot(t, y[1], label='y_1')
    ##new Animation
    plt.legend()
    fig = plt.figure()
    Floors, = plt.plot([], [], 'go', ms=3)
    plt.xlim([-np.max(np.abs(x[len(M):,:])),np.max(np.abs(x[len(M):,:]))])
    plt.ylim([-1, len(M)])
    ani = animation.FuncAnimation(fig, animate,frames=60,interval=1, blit=False, init_func=init)
    plt.show()
    print(t[20], x[len(M):,20])
    #print(Model)
