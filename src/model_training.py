import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error

def matrix_factorization(R_mat, k=2):
    """
    Perform matrix factorization on user-item matrix
    using Singular Value Decomposition (SVD)
    
    Args: 
        R: user-item matrix
        k: number of reserved components

    Output: Return matrix-factorized user-item matrix
    """
    U, sigma, Vt = svds(R_mat, k=k)
    R_pred = np.dot(np.dot(U, np.diag(sigma)), Vt)
    print("Done Matrix Factorization on R.")
    return R_pred

def item_based_CF_model(R, n_neighbors=30):
    knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    knn.fit(R.T)

    print("Done fitting item_based_CF_model.")
    return knn

def item_content_based_model(item_content, n_neighbors=30):
    knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    knn.fit(item_content)

    print("Done fitting item_content_based_model.")
    return knn

def user_content_based_model(user_content, n_neighbors=30):
    """
    Return KNN model fitted on user_content matrix
    predicting similar users with content-based approach
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors+1, metric="cosine")
    knn.fit(user_content)

    print("Done fitting user_content_based_model.")
    return knn
