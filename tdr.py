import numpy as np
from copy import deepcopy

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

def build_tdr(X,N):
    """
    Perform TDR analyses as in Sun, O'Shea et al, 2021.

    Parameters:
    - X (ndarray): Behavioral variables before learning (C by M matrix)
    - N (ndarray): Condition-averaged, centered neural activity before learning (C by N matrix)

    Returns:
    - beta_n2b_orth (ndarray): Matrix of orthogonalized coefficients projecting neural activity to TDR axes
    """
    # Get ready the design matrix (behavioral variables + intercept).
    X = np.hstack((X,np.ones((X.shape[0],1))))

    # Regress neural data against the design matrix and compute the regression coefficients.
    beta_b2n = np.linalg.pinv(X) @ N
    #beta_b2n = np.linalg.lstsq(X, N, rcond=None)[0]

    # Compute the TDR axes.
    beta_n2b = np.linalg.pinv(beta_b2n)
    beta_n2b = beta_n2b[:, :2]

    # Orthogonalize the TDR axes before projection.
    beta_n2b_orth = gsog(beta_n2b)[0]

    return beta_n2b_orth

def project_onto_map(data,map,remove_mean=True):
    """
    Returns:
    - data_p (ndarray): Matrix of neural state coordinates on orthogonalized TDR axes
    """
    data_p = deepcopy(data)
    # remove the mean
    combined_N = np.vstack(data)
    mean_N = np.mean(combined_N, axis=0)

    if remove_mean==False:
        mean_N = np.zeros_like(mean_N)

    for i in range(len(data)):
        data_p[i] = (data[i]-mean_N) @ map
    return data_p

def orth_wrt_map(us, map):
    us_orth = us.copy()  # Start with the original vector
    for i in range(map.shape[1]):  # Iterate over each column of the map
        # Project us onto the current column and subtract the projection from us_orth
        us_orth = us_orth - np.dot(map[:,i], us_orth)/np.linalg.norm(map[:,i])**2 * map[:,i][:,None]
    us_orth_norm = us_orth / np.linalg.norm(us_orth)  # Normalize the orthogonal vector
    return us_orth_norm

