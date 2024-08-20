from src.utils import create_mapper, MatrixPackage
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")


def create_user_item_matrix(interactions):
    """
    Create user-item matrix from user-item-rating data
    Return sparse csr_matrix user-item matrix and its mappers
    """
    customer_mapper, customer_inv_mapper, M = create_mapper(interactions["customer_id"])
    article_mapper, article_inv_mapper, N = create_mapper(interactions["article_id"])

    ratings = interactions["score"]
    row_ind = [customer_mapper[i] for i in interactions["customer_id"]]
    col_ind = [article_mapper[i] for i in interactions["article_id"]]

    R = csr_matrix((ratings, (row_ind, col_ind)), shape=(M,N))

    print("Done creating user-item matrix.")
    return MatrixPackage(
        R, 
        {"row": customer_mapper, "col": article_mapper}, 
        {"row": customer_inv_mapper, "col": article_inv_mapper}, 
        (M,N)
    )

    
def reduced_article_content(articles, n_components=30):
    """
    Turn articles descriptions into bag-of-words feature,
    using svd for dimensionality reduction
    Return matrix of article reduced information, feature_mapper, feature_inv_mapper
    """
    articles = articles.sort_values(by="article_id")
    mapper, inv_mapper, length = create_mapper(articles["article_id"])
    
    vect = TfidfVectorizer()
    X = vect.fit_transform(articles["desc"])

    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    
    print("Done creating article content matrix.")
    return MatrixPackage(X_reduced, {"row": mapper}, {"row": inv_mapper}, X_reduced.shape)

def customer_content(customers):
    """
    Turn customers dataset into matrix
    Return customer-matrix, info_mapper, info_inv_mapper
    """
    customers = customers.sort_values(by="customer_id")
    mapper, inv_mapper, length = create_mapper(customers["customer_id"])
    
    X = customers.drop(columns=["customer_id"]).values

    print("Done creating customer content matrix.")
    return MatrixPackage(X, {"row": mapper}, {"row": inv_mapper}, X.shape)



