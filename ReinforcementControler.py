import numpy as np
import pandas as pd
import random
import control as con
import matplotlib.pyplot as plt
from numba import jit
import matplotlib.animation as animation
import scipy.io as sio
import gym
from scores.score_logger import ScoreLogger
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def LoadSeismicData(Earthquake, extension):
    path="prueba_sistema_discreto/inputs/"+Earthquake+"/"+Earthquake+extension
    mat_contents = sio.loadmat(path)
    return mat_contents["elcentro_NS"],mat_contents["time_NS"]

def SaveModel(ModelName):
    masses=np.array([6,6,6])*1e3 #mass[kg]
    stiffness=np.array([1.8,1.6,1.6])*1e6# [N/m]
    dampings=np.array([7.24,5.16,7.20])*1e3 # [N s/m]
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
    Inter=np.zeros((n,n))
    for i in range(n):
        if(i==0):
            Inter[i,i]=1
        else:
            Inter[i,i]=1
            Inter[i,i-1]=-1
    print(Inter)
    Drifting=np.block([[np.zeros((n,n)),np.ones((n,n))],[np.zeros((n,n)),Inter]])
    C= np.ones((1,2*n))@Drifting
    sys = con.StateSpace(A, B, C, D)
    return sys
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

def sim_step(State, Model, Input, time):
    terminal= False
    print("time",time)
    State = np.reshape(State, [observation_space,1])
    t, y, x = con.forced_response(Model, U=Input, T=time, X0=State)
    print("score:",1/np.linalg.norm(y))
    if (np.linalg.norm(y)>0.1):
        terminal=True
    return (x[:,-1], 1/np.linalg.norm(y), terminal)

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
    Floors.set_data(x[9:,m%n[1]], range(int(n[0]/2)))
    return Floors
if __name__ == "__main__":

    #We define the learning parameters
    GAMMA = 0.95
    LEARNING_RATE = 0.001

    MEMORY_SIZE = 1000000
    BATCH_SIZE = 20

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    #We define the seismic activity
    global m
    m=0


    ##Choose Any earthquake from the list
    EQ=["elcentro","chi-chi","Kobe"]
    Ext=["_NS.mat","_NS.mat","_NS.mat"]
    W_t,t=LoadSeismicData(EQ[0], Ext[0])
    U_t=np.block([[np.zeros((1,len(t)))],[W_t.T*9.8]])


    #we define the building paramters
    ##Choose Any model of the list
    #SaveModel("ThreeFloors")
    models=["NineFloors", "ThreeFloors","FiveFloors"]
    M,S,D=LoadDatamodel(models[1])
    Model=GenerateModel(M,S,D)


    #We create a script to store the perfomance data
    score_logger = ScoreLogger("Active_Structure")


    #Now We initialize the model
    #print("Model:",Model.A.shape[1])
    observation_space = Model.A.shape[0]
    #we are considering just one magneto damper
    action_space = 2
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        # Reset
        state = np.zeros((observation_space,1))
        state = np.reshape(state, [1, observation_space])
        step = 0
        timeaverage=0
        score= 0
        last_U=np.block([[0],[W_t[0]*9.8]])
        #t=0
        while True:
            step += 1
            timeaverage+= 2
            #env.render()
            action = dqn_solver.act(state)
            print([(action-0.5)*2*0.1])
            U=np.block([[(action-0.5)*2*0.01],[W_t[step-1]*9.8]])
            U_T=np.block([U,U])
            state_next, reward, terminal = sim_step(state, Model, U_T, t[(timeaverage-1)%len(W_t):timeaverage+1%len(W_t)].T)
            reward = reward if not terminal else -100
            score+=reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            last_U=U
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(score)+", step: " + str(step))
                score_logger.add_score(int(score), run)
                break
            dqn_solver.experience_replay()
            #state = np.reshape(state, [observation_space,1])

    #t, y, x = con.forced_response(Model, U=U_t, T=t.T)

    '''
    for j in range(len(M)):
        plt.plot(t[:300], x[9+j,:300], label="Floor: "+str(j))
    #plt.plot(t, y[1], label='y_1')
    ##new Animation
    plt.legend()
    fig = plt.figure()
    Floors, = plt.plot([], [], 'go', ms=3)
    plt.xlim([-np.max(np.abs(x[9:,:])),np.max(np.abs(x[9:,:]))])
    plt.ylim([-1, len(M)])
    ani = animation.FuncAnimation(fig, animate,frames=60,interval=1, blit=False, init_func=init)
    plt.show()
    print(t[20], x[9:,20])
    '''
    #print(Model)
