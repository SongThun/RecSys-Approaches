## Recommender System H&M
### Overview
This is a learning repository, exploring basic topics in Recommender System including Collaborative Filtering and Content-based Approach.
The project uses [H&M RecSys Young Female Data](https://www.kaggle.com/datasets/zhenglinkevinwang/hm-recsys-young-female-data).

### Features:
- Data processing and Feature engineering
- Collaborative Filtering: basic item-based recommendation, Matrix factorization (SVD)
- Content-based: user and item features, user profile with item features

### Getting started
#### Installation
1. Clone the repository
```
git clone https://github.com/SongThun/RecSys-Approaches.git
cd RecSys-Approaches
```
2. Install dependencies
```
pip install -r requirements.txt
```
#### Running the project
1. Download and unzip the dataset from the link above to `data/raw`
2. Data processing and Model training
```
py scripts/train.py
```
3. Generate recommendations:
```
py scripts/test.py --customer_id <CUSTOMER_ID> --approach <APPROACH> [--n_recs <NUMBER_OF_RECOMMENDATIONS>] [--display]
```

### Usage - Generate recommendations
- `--customer_id`: integer number indicate the mapped index of customer_id (not the real customer_id)
- `--approach`:
  + `'cf'`: collaborative filtering on user-item matrix
  + `'cfmf'`: collaborative filtering using matrix factorization
  + `'cbi`: content-based using item (article) features
  + `'cbu'`: content-based using user (customer) features
  + `'cbup'`: combine item features and user-item matrix to form user profiles
- [optional] `--n_recs`: number of recommendations (default: 5)
- [optional] `--display`: display the actual customer_id and their most recent transactions
