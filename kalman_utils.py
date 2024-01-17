import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter


def pred_state(behav,neural,M1,M2):

    X=np.matrix(behav.T)
    Z=np.matrix(neural.T)

    num_states=X.shape[0] #Dimensionality of the state

    states=np.empty(X.shape)
    state=X[:,0] #Initial state
    states[:,0]=np.copy(np.squeeze(state))


    for t in range(X.shape[1]-1):

        state=M1*state+M2*Z[:,t+1]
        states[:,t+1]=np.squeeze(state) #Record state at the timestep
    
    return states.T

def find_M2(behav,neural,M1):
    Z = np.matrix(neural.T)
    Z2 = Z[:,1:]

    X = np.matrix(behav.T)
    X2 = X[:,1:]
    X1 = X[:,:-1]
    
    dX = X2-M1*X1
    M2 = dX*Z2.T*inv(Z2*Z2.T)

    return M2
