import numpy as np
import copy

class MatrixPackage:
    def __init__(self, mat, mappers, inv_mappers, shape):
        if mat.shape != shape: 
            raise(f"Inconsistent shape: mat.shape = {mat.shape}, shape = {shape}")
        if len(mappers) == 0:
            raise(f"No mappers")
        if len(inv_mappers) == 0:
            raise(f"No inv_mappers")
        
        self.mat = mat
        self.mappers = mappers
        self.inv_mappers = inv_mappers
        self.shape = shape

    def row_map(self, id):
        return self.mappers["row"][id]
    
    def col_map(self, id):
        if "col" in self.mappers:
            return self.mappers["col"][id]
        raise("No column mapper")
    
    def row_inv_map(self, index):
        return self.inv_mappers["row"][index]
    
    def col_inv_map(self, index):
        if "col" in self.inv_mappers:
            return self.inv_mappers["col"][index]
        raise("No column inverse mapper.")
        
    
def create_mapper(identifier):
    """
    Create mapping for pd.Series of identifier, sorted lexicographically
    
    Args: identifer - pd.Series or dataframe column
    
    Output: 
        mapper: identifier -> index
        inv_mapper: index -> identifier    
    """

    length = identifier.nunique()

    mapper = dict(zip(np.unique(identifier), list(range(length))))
    inv_mapper = dict(zip(list(range(length)), np.unique(identifier)))

    return mapper, inv_mapper, length


def normalize(scores, range=(1,10)):
    """
    Normalize pd.Series `scores` to fall within `range`
    """
    scaled = (scores - scores.min()) / (scores.max() - scores.min())
    return scaled * (range[1] - range[0]) + range[0]

def display_articles(datasets, article_list):
    """
    Print the information from `article_info` dataset of articles in `article_list`
    """
    indices = datasets["article_id"].isin(article_list)

    print(datasets[indices].to_string(index=False))