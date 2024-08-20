import os
from src.utils import MatrixPackage
from src.data_preprocessing import *
from src.feature_engineering import *
from src.model_training import *
from src.recommendation import *
import joblib

def main():
    # data preprocessing
    data_path = "./data"

    datasets = load_data(data_path)
    clean_datasets = clean_data(datasets)

    for key in clean_datasets:
        save_dataset(clean_datasets[key], f"{data_path}/processed/{key}.csv")

    # create required matrices
    interactions = create_interaction_data(clean_datasets["transactions"])
    R = create_user_item_matrix(interactions)
    X_articles = reduced_article_content(clean_datasets["articles"])
    X_customers = customer_content(clean_datasets["customers"])
    
    joblib.dump(R, "./models/variables/R.pkl")
    joblib.dump(X_articles, "./models/variables/X_articles.pkl")
    joblib.dump(X_customers, "./models/variables/X_customers.pkl")
    
    # collaborative filtering
    R_pred = matrix_factorization(R.mat)
    R_pred_pack = MatrixPackage(R_pred, R.mappers, R.inv_mappers, R.shape)
    joblib.dump(R_pred_pack, "./models/variables/R_pred.pkl")
    
    cf_item_model = item_based_CF_model(R.mat)
    cb_item_model = item_content_based_model(X_articles.mat)
    cb_user_model = user_content_based_model(X_customers.mat)

    joblib.dump(cf_item_model, "./models/cf_item_model.pkl")
    joblib.dump(cb_item_model, './models/cb_item_model.pkl')
    joblib.dump(cb_user_model, "./models/cb_user_model.pkl")

    print("Training finished.")
    return 0

if __name__ == "__main__":
    main()
