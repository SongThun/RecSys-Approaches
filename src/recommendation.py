import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_articles(article_id, M, model, approach="cf", n_recs=10):
    """
    Find similar items by collaborative filtering
    Args:
        article_id
        M:  matrix of articles, choose between
            'cf': collaborative filtering approach
            'content': content-based by item approach
            If approach == "cf" then it should be a user-item packages.
            
        approach:
        n_recs: number of recommendations

    Return: list of similar article_ids
    """
    mat = M.mat

    if approach == "cf":
        article_mapper = M.col_map
        article_inv_mapper = M.col_inv_map
        mat = mat.T
    elif approach == "content":
        article_mapper = M.row_map
        article_inv_mapper = M.row_inv_map
    else:
        raise(f"No approach option {approach}. Please choose between ['cf', 'content']")
    
    aid = article_mapper(article_id)
    article = mat[aid].toarray().flatten() if approach == "cf" else mat[aid]
    neighbors = model.kneighbors([article], return_distance=False)[0]

    return [article_inv_mapper(i) for i in neighbors if i != aid][:n_recs]

def recommendation_by_MF(customer_id, R, R_pred, n_recs=10):
    """
    Make recommendation with the matrix resulting from matrix factorization
    """
    cid = R.row_map(customer_id)
    customer = R.mat[cid].toarray().flatten()
    purchased = np.where(customer > 0)[0]

    customer_pred = R_pred.mat[cid]
    scores = customer_pred.argsort()[::-1]
    top_scores = [i for i in scores if i not in purchased]

    return [R.col_inv_map(i) for i in top_scores][:n_recs]


def customer_profile(customer_id, R, X):
    """
    Build customer profile from transaction history
    """
    customer = R.mat[R.row_map(customer_id)].toarray().flatten()
    purchased = np.where(customer > 0)[0]
    feature_indices = [X.row_map(R.col_inv_map(i)) for i in purchased]
    scores = np.matmul(customer[purchased], X.mat[feature_indices])
    return scores

def recommendation_by_user_profile(customer_id, X, R, model, n_recs=10):
    """
    Make recommendation based on customer transaction history
    Args:
        customer_id
        X: item-content package
        R: user-item package
        n_recs: number of recommendations
    """

    cid = R.row_map(customer_id)
    
    profile = customer_profile(customer_id, R, X)
    neighbors = model.kneighbors([profile], return_distance=False)[0]

    purchased = np.where(R.mat[cid].toarray().flatten() > 0)[0]
    purchased_feature_ids = [X.row_map(R.col_inv_map(i))
                            for i in purchased]
    
    top_N_indices = [i for i in neighbors if i not in purchased_feature_ids]
    
    return [X.row_inv_map(i) for i in top_N_indices][:n_recs]



def recommendation_by_similar_customers(customer_id, X, R, model, n_recs=10):
    """
    Make recommendations by content-based approach on user-content
    Args:
        cid: mapped index of customer_id on X
        X: user-content packages
        R: user-item packages
        model: KNN model fitted on X
        n_recs: number of recommendations

    Return: list of n_recs article_ids
    """
    
    # take the neighborhood and their corresponding distance
    cid = X.row_map(customer_id)
    distance, neighbors = model.kneighbors([X.mat[cid]])
    mask = neighbors[0] != cid
    neighbors = neighbors[0][mask]
    cosine = (1 - distance[0])[mask]

    # extract similar customers scorings
    active = R.mat[R.row_map(customer_id)].toarray().flatten()
    similar_cust_ids = [R.row_map(X.row_inv_map(i)) for i in neighbors]
    similar_cust_R = R.mat[similar_cust_ids].toarray()
    similar_cust_mean = similar_cust_R.mean(axis=1).reshape(-1,1)
    
    # calculate predicted articles scoring of the active user
    scores = (
        active.mean() 
        + np.dot(cosine, similar_cust_R - similar_cust_mean) 
        / cosine.sum()
    )
    scores = scores.argsort()[::-1]
    
    # exclude those already purchased
    purchased = np.where(active > 0)[0]
    
    return [R.col_inv_map(i) for i in scores if i not in purchased][:n_recs]
