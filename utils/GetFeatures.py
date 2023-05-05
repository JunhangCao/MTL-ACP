import numpy as np
import pandas as pd
from utils.physicochemical import pc_properties

"""
    If there are new amino acid sequences which are not in our provided dataset, 
    this module is able to get the physico-chemical properties for you.
"""


def load_file(path, tag):
    """
    :param path:
    :param tag: must be NAMP、ABP、AFP、AHP、AVP、ACP
    :return:
    """
    sequences = []
    with open(path, 'r') as f:
        file = f.readlines()
        for i, s in enumerate(file):
            if s[0] != '>':
                sequences.append(s[:-1])

    if tag == 'NAMP':
        target = [[1, 0, 0, 0, 0, 0] for _ in range(len(sequences))]
    elif tag == 'ABP':
        target = [[0, 1, 0, 0, 0, 0] for _ in range(len(sequences))]
    elif tag == 'AFP':
        target = [[0, 0, 1, 0, 0, 0] for _ in range(len(sequences))]
    elif tag == 'AHP':
        target = [[0, 0, 0, 1, 0, 0] for _ in range(len(sequences))]
    elif tag == 'AVP':
        target = [[0, 0, 0, 0, 1, 0] for _ in range(len(sequences))]
    elif tag == 'ACP':
        target = [[0, 0, 0, 0, 0, 1] for _ in range(len(sequences))]
    else:
        raise ValueError
    return sequences, target


def get_features(path, tag):
    sequences, target = load_file(path, tag)
    features, target, features_df = pc_properties(sequences, target)
    # save features as csv file
    # features_df.save('./features/'+tag+'.csv')

    return features, target
