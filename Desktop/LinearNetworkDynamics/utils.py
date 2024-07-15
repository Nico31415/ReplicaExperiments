import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.decomposition import PCA

def reshape_matrix(input_matrix, new_shape):

    old_shape = input_matrix.shape
    

    if new_shape[0] > old_shape[0]:

        new_matrix = np.vstack((input_matrix, np.zeros((new_shape[0] - old_shape[0], old_shape[1]))))
    elif new_shape[0] < old_shape[0]:

        new_matrix = input_matrix[:new_shape[0], :]
    else:
        new_matrix = input_matrix
    
    if new_shape[1] > old_shape[1]:
        new_matrix = np.hstack((new_matrix, np.zeros((new_shape[0], new_shape[1] - old_shape[1]))))
    elif new_shape[1] < old_shape[1]:
        new_matrix = new_matrix[:, :new_shape[1]]
    
    return new_matrix

def balanced_weights(in_dim, hidden_dim, out_dim, sigma = 1):
    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    w1 = sigma * np.random.randn(hidden_dim, in_dim)
    w2 = sigma * np.random.randn(out_dim, hidden_dim)

    U_, S_, V_ = np.linalg.svd(w2 @ w1)
    s = np.sqrt(np.diag(S_))

    lmda = np.trace(w2 @ w1) / hidden_dim

    factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

    s_2 = np.sqrt(np.diag(np.diag(factor)))

    s2_reshaped = reshape_matrix(s_2, (out_dim, hidden_dim))

    s_1 = np.diag(np.diag(s) / np.diag(s_2))

    s1_reshaped = reshape_matrix(s_1, (hidden_dim, in_dim))

    S_test = s2_reshaped @ s1_reshaped

    w1_out = r @ s1_reshaped @ V_.T 

    w2_out = U_ @ s2_reshaped @ r.T

    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    scale_by = lmda / q[0][0]
    w1_out = scale_by * w1_out
    w2_out = scale_by * w2_out
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    return w1_out, w2_out, S_test, q


def get_lambda_balanced_aligned(lmda, in_dim, hidden_dim, out_dim, X, Y, sigma=1):
    U, _, Vt = np.linalg.svd(Y @ X.T)

    w1, w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)
    U_, _, Vt_ = np.linalg.svd(w2 @ w1)

    init_w2 = U @ U_.T @ w2 
    init_w1 = w1 @ Vt_.T @ Vt

    return init_w1, init_w2


def get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim, sigma=1):

    if out_dim > in_dim and lmda < 0:
        print('Lambda must be positive if out_dim > in_dim')
        return 
    if in_dim > out_dim and lmda > 0:
        print('Lambda must be positive if in_dim > out_dim')
        return 
    if hidden_dim < min(in_dim, out_dim):
        print('Network cannot be bottlenecked')
        return 
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return 
    
    #add check here for dimensions and lambda
    w1 = sigma * np.random.randn(hidden_dim, in_dim)
    w2 = sigma * np.random.randn(out_dim, hidden_dim)

    U, S, Vt = np.linalg.svd(w2 @ w1)

    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))

    S2_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) + lmda) / 2))
    S1_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) - lmda) / 2))

    if out_dim > in_dim:
        add_terms = np.asarray([np.sqrt(lmda) for _ in range(hidden_dim - in_dim)])
        S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
                        np.zeros((out_dim - hidden_dim, hidden_dim))]) 
        S1 = np.vstack([np.diag(S1_equal_dim), 
                        np.zeros((hidden_dim - in_dim, in_dim))])
    elif in_dim > out_dim:
        add_terms = np.asarray([-np.sqrt(-lmda) for _ in range(hidden_dim-out_dim)])
        S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
                        np.zeros((hidden_dim, in_dim - hidden_dim))])
        S2 = np.hstack([np.diag(S2_equal_dim), 
                        np.zeros((out_dim, hidden_dim - out_dim))]) 
        
    else:
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)
    init_w2 =  U @ S2 @ R.T 
    init_w1 = R @ S1 @ Vt 

    return init_w1, init_w2 

def whiten(X):

    scaler = StandardScaler()

    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    # print('explained variance: ', pca.explained_variance_)
    X_whitened = X_pca / np.sqrt(pca.explained_variance_)

    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened

    return X_whitened

def get_random_regression_task(batch_size, in_dim, out_dim):
    X = np.random.randn(batch_size, in_dim)
    Y = np.random.randn(batch_size, out_dim)
    X_whitened = whiten(X)

    return X_whitened.T, Y.T

# def whiten(X):

#     scaler = StandardScaler()
#     X_standardised = scaler.fit_transform(X)

#     # print('X standardised: ', X_standardised)
    
#     pca = PCA()
#     X_pca = pca.fit_transform(X_standardised)

#     X_whitened = torch.tensor(X_pca / np.sqrt(pca.explained_variance_), dtype=torch.float32)

#     return X_whitened

# def get_random_regression_task(batch_size, in_dim, out_dim):
#     X = torch.randn(batch_size, in_dim)
#     print(X)
#     Y = torch.randn(batch_size, out_dim)
#     print(Y)
#     X_whitened = whiten(X)

#     return X_whitened.clone().T, Y.clone().T

class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass