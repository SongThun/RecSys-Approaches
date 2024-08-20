import pandas as pd
import argparse
import joblib
from src.recommendation import *
from src.data_preprocessing import find_most_recent_articles
from src.utils import display_articles

# Load models and data once
articles = pd.read_csv("./data/processed/articles.csv")
transactions = pd.read_csv("./data/processed/transactions.csv")
R = joblib.load("./models/variables/R.pkl")
R_pred = joblib.load("./models/variables/R_pred.pkl")
X_articles = joblib.load("./models/variables/X_articles.pkl")
X_customers = joblib.load("./models/variables/X_customers.pkl")

cf_item_model = joblib.load("./models/cf_item_model.pkl")
cb_item_model = joblib.load('./models/cb_item_model.pkl')
cb_user_model = joblib.load("./models/cb_user_model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action="store_true", help="Print customer recent transaction history")
    parser.add_argument('--customer_id', type=int, help="Mapped index for a customer_id")
    parser.add_argument('--approach', type=str, help="Recommender system approach")
    parser.add_argument('--n_recs', type=int, help="Number of recommendations.")

    args = parser.parse_args()

    n_recs = 5 if args.n_recs is None or args.n_recs > 5 else args.n_recs

    if args.customer_id is None or args.approach is None:
        print("Must provide --customer_id and --approach")
        return
    
    cid = X_customers.row_inv_map(args.customer_id)
    aid, n_aids = find_most_recent_articles(transactions, cid)
    if args.display is not None:
        display_articles(articles, n_aids)
    match (args.approach):
        case "cf":
            print("[CF] Recommendation by most recently purchased item:")
            display_articles(articles, find_similar_articles(aid, R, cf_item_model, n_recs=n_recs))
        case "cfmf":
            print("[CF] Recommendation by matrix factorization:")
            display_articles(articles, recommendation_by_MF(cid, R, R_pred, n_recs))
        case "cbi":
            print("[CB] Recomendation by most recent purchased item:")
            display_articles(articles, find_similar_articles(aid, X_articles, cb_item_model, approach="content", n_recs=n_recs))
        case "cbu":
            print("[CB] Recommendation by similar users:")
            display_articles(articles, recommendation_by_similar_customers(cid, X_customers, R, cb_user_model, n_recs))
        case "cbup":
            print("[Hybrid] Recommendation by customer profile:")
            display_articles(articles, recommendation_by_user_profile(cid, X_articles, R, cb_item_model, n_recs))
        case _:
            print(f"No approach names {args.approach}")
            print("Please choose amongst ['cf', 'cfmf', 'cbi', 'cbu', 'cbup']")
            return


if __name__ == "__main__":
    main()