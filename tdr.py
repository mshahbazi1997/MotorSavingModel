import numpy as np
from sklearn.decomposition import PCA

def get_data(data,phases,go_cue_idx=10,force_idx=22,index_neuron=None):
    """
    data: list of data
    phases: list of phases

    return:
    Data_prep: dict
        X: list of muscle force at peak velocity
        X_ldim: list of low dim muscle force at peak velocity
        Y: list of neural activity at go cue
    """

    # get_data
    Data_prep =  {'X':[],'X_ldim':[],'Y':[],'Y_centered':[]}


    for i in range(len(phases)):
        
        # behavior
        vel = np.array(data[i]['vel'])
    
        vel_norm = np.linalg.norm(vel,axis=-1)
        max_vel_idx = np.argmax(vel_norm,axis=1)
        max_vel_idx = np.zeros_like(max_vel_idx)
        max_vel_idx[:] = force_idx

        #muscle_force = np.array(data[i]['all_force'])
        #n_cond = muscle_force.shape[0]

        #muscle_force_at_peak_vel = muscle_force[np.arange(n_cond), max_vel_idx, :]
        #Data_prep['X'].append(muscle_force_at_peak_vel)
        

        # low d behavior
        # TODO: need to for the pca over all phases...
        #pca = PCA(n_components=6)
        #X_pca = pca.fit_transform(muscle_force_at_peak_vel)
        #X_pca = X_pca[:, :2]
        #Data_prep['X_ldim'].append(X_pca)


        cartesian_force = np.array(data[i]['endpoint_force'])
        n_cond = cartesian_force.shape[0]
        cartesian_force_at_peak_vel = cartesian_force[np.arange(n_cond), max_vel_idx, :]
        Data_prep['X'].append(cartesian_force_at_peak_vel)

        #dT = 0.01
        #acc = np.diff(vel,axis=1)/dT
        #acc_at_peak_vel = acc[np.arange(n_cond), max_vel_idx, :]
        #Data_prep['X'].append(acc_at_peak_vel)
        #Data_prep['X_ldim'].append(acc_at_peak_vel)
        
        # neural
        go_cue_idx = go_cue_idx
        if index_neuron is not None:
            fr = np.array(data[i]['all_hidden'][:,:,index_neuron])
        else:
            fr = np.array(data[i]['all_hidden'])
        fr_at_go_cue = fr[:, go_cue_idx, :]
        Data_prep['Y'].append(fr_at_go_cue)

    return Data_prep


def gsog(X):
    """
    Gram-Schmidt orthogonalization

    Parameters:
    - X (ndarray): Input matrix of size (d, n)

    Returns:
    - Q (ndarray): Orthogonalized matrix of size (d, m)
    - R (ndarray): Upper triangular matrix of size (m, n)
    """

    d, n = X.shape
    m = min(d, n)

    R = np.eye(m, n)
    Q = np.zeros((d, m))
    D = np.zeros(m)

    for i in range(m):
        R[0:i, i] = np.dot(np.multiply(Q[:, 0:i], 1 / D[0:i]).T, X[:, i])
        Q[:, i] = X[:, i] - np.dot(Q[:, 0:i], R[0:i, i])
        D[i] = np.dot(Q[:, i], Q[:, i])

    R[:, m:n] = np.dot(np.multiply(Q, 1 / D).T, X[:, m:n])

    return Q, R


def build_TDR_subspace(before_learning_X, before_learning_N, after_learning_X , after_learning_N=None, TDR_options=1):
    """
    Perform TDR analyses as in Sun, O'Shea et al, 2021.

    Parameters:
    - before_learning_X (ndarray): Behavioral variables before learning (C by M matrix)
    - after_learning_X (ndarray): Behavioral variables after learning (C by M matrix)
    - before_learning_N (ndarray): Condition-averaged, centered neural activity before learning (C by N matrix)
    - after_learning_N (ndarray): Condition-averaged, centered neural activity after learning (C by N matrix)
    - TDR_options (int): Indicator of whether to use before-learning data (1), after-learning data (2), or both (3)

    Returns:
    - betaNeural2Behav (ndarray): Matrix of un-orthogonalized coefficients projecting neural activity to TDR axes
    - betaNeural2BehavOrth (ndarray): Matrix of orthogonalized coefficients projecting neural activity to TDR axes
    - projectedStates (ndarray): Matrix of neural state coordinates on orthogonalized TDR axes
    """

    # Now build the TDR subspace
    if TDR_options == 1:
        X = before_learning_X
        N = before_learning_N
    elif TDR_options == 2:
        X = after_learning_X
        N = after_learning_N
    elif TDR_options == 3:
        X = np.vstack((before_learning_X, after_learning_X))
        N = np.vstack((before_learning_N, after_learning_N))
    else:
        raise ValueError("Invalid value for TDR_options. Choose 1, 2, or 3.")

    # Get ready the design matrix (behavioral variables + intercept).
    behav = np.hstack((X, np.ones((X.shape[0], 1))))

    # Regress neural data against the design matrix and compute the regression coefficients.
    betaBehav2Neural = np.linalg.lstsq(behav, N, rcond=None)[0]

    # Compute the TDR axes.
    betaNeural2Behav = np.linalg.pinv(betaBehav2Neural)
    betaNeural2Behav = betaNeural2Behav[:, :X.shape[1]]

    # Orthogonalize the TDR axes before projection.
    betaNeural2BehavOrth = gsog(betaNeural2Behav)

    # Project before-learning and after-learning neural activity onto the TDR axes
    projectedStates = np.vstack((before_learning_N, after_learning_N)) @ betaNeural2BehavOrth[0]

    return betaNeural2Behav, betaNeural2BehavOrth[0], projectedStates


