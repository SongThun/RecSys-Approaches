import os
import pandas as pd
import numpy as np
from src.utils import normalize
import warnings

warnings.filterwarnings("ignore")

def save_dataset(dataset, save_path):
    """
    Save pd.DataFrame `dataset` into `save_path` - path/to/dataset.csv
    """
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Deleted old file: {save_path}")

    dataset.to_csv(save_path)
    print(f"File created: {save_path}")


def load_data(data_path):
    datasets = {}
    datasets["transactions"] = pd.read_csv(f"{data_path}/raw/young_female_trans.csv")
    datasets["customers"] = pd.read_csv(f"{data_path}/raw/customers.csv")
    datasets["articles"] = pd.read_csv(f"{data_path}/raw/articles.csv")
    return datasets

def preprocess_articles(articles):
    """
    Join articles' text columns into one description
    
    Args: articles - pd.DataFrame of articles.csv

    Output: cleaned dataset with columns ["article_id", "desc"]
    """
    articles_text = articles.filter(regex='name$|detail_desc')
    articles_text = articles_text.fillna('')
    descriptions = articles_text.agg(' '.join, axis=1)

    descriptions =  pd.concat([articles["article_id"], descriptions], axis=1)
    descriptions.columns = ["article_id", "desc"]
    
    return descriptions


def preprocess_transactions(transactions):
    """
    Aggregate the information of `transactions` dataset and store in `save_path`

    Args: transactions - pd.DataFrame of young_female_trans.csv

    Output: the aggregated dataset 
    """
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["date_diff"] = (pd.to_datetime("today") - transactions["t_dat"]).dt.days

    group = ["customer_id", "article_id", "t_dat"]
    trans_stat = transactions.groupby(group)["price"].agg(["count", "sum"]).reset_index()

    agg_trans = pd.merge(trans_stat, transactions, on=group)
    
    return agg_trans


def preprocess_customers(customers, transactions):
    """
    Extracting only young active female customers,
    making new features from transaction history
    
    Args: 
        customers - pd.DataFrame of customers.csv
        transactions - pd.DataFrame of aggregated transactions dataset

    Output: processed young female customers dataset
    """

    # take only young female customers
    young_customer_ids = np.unique(transactions["customer_id"])
    young_customers = customers[customers["customer_id"].isin(young_customer_ids)]
    young_customers = young_customers[["customer_id", "age"]]

    # calculate avg spending per month
    customer_spent = transactions.copy()[["customer_id", "article_id", "t_dat", "sum", "sales_channel_id"]]
    customer_spent["t_dat"] = pd.to_datetime(customer_spent["t_dat"])
    customer_spent["month"] = customer_spent["t_dat"].dt.to_period('M')

    customer_total_spent = customer_spent.groupby("customer_id")["sum"].sum().reset_index()
    customer_total_months = customer_spent.groupby("customer_id")["month"].nunique().reset_index()
    
    customer_info = pd.merge(customer_total_spent, customer_total_months)
    customer_info["avg_per_month"] = customer_info["sum"] / customer_info["month"]


    # take the most active channel for each user
    mode = lambda x : x.mode().iloc[0] if not x.mode().empty else None
    customer_most_active_channel = customer_spent.groupby("customer_id")["sales_channel_id"].agg(mode).reset_index()


    # finalize the dataset 
    customer_info = pd.merge(customer_info, customer_most_active_channel)
    customer_info = pd.merge(customer_info, young_customers)

    return customer_info
    

def clean_data(datasets):
    articles = preprocess_articles(datasets["articles"])
    transactions = preprocess_transactions(datasets["transactions"])
    customers = preprocess_customers(datasets["customers"], transactions)
    return {
        "articles": articles, 
        "transactions": transactions, 
        "customers": customers
    }

def create_interaction_data(transactions):
    """
    Turn transactions data into user-item-rating format
    Return dataset of the form ["customer_id", "article_id", "score"] 
            with `score` normalized to [1-10] range 
    """

    transactions['score_by_date'] = transactions['count'] / (transactions['date_diff'] + 1)
    transaction_scores = transactions.groupby(['customer_id', 'article_id'])['score_by_date'].agg('sum').reset_index(name='score')

    # cap score to 90-percentile
    q1 = transaction_scores['score'].quantile(0.25)
    q3 = transaction_scores['score'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + iqr * 1.5

    transaction_scores['score'] = transaction_scores['score'].apply(lambda x: upper if x > upper else x)
    transaction_scores['score'] = normalize(transaction_scores['score'], range=(1,10))

    return transaction_scores

def find_most_recent_articles(datasets, customer_id, n_recent=10):
    """
    Return `n-recent` most-recently-purchased articles by customer
    """ 
    indices = datasets["customer_id"] == customer_id
    transaction_history = datasets[indices].sort_values(by="t_dat", ascending=False)
    articles_history = transaction_history["article_id"]
    top_recent = articles_history[:n_recent]
    most_recent = articles_history.iloc[0]

    return most_recent, top_recent