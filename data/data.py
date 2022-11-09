"""
    File to load dataset based on user control from main file
"""
from data.SBMs import SBMsDataset
from dgl.data import CoraGraphDataset, CiteSeerDataset, RedditDataset, PubmedGraphDataset

def load_data(dataset):
    """
        This function is called in the main_xx.py file
        returns:
        ; dataset object
    """
    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if dataset in SBM_DATASETS:
        return SBMsDataset(dataset)
    elif dataset == 'CORA':
        return CoraGraphDataset()
    elif dataset == 'CITESEER':
        return CiteSeerDataset()
    elif dataset == 'REDDIT':
        return RedditDataset()
    elif dataset == 'PUBMED':
        return PubmedGraphDataset()


