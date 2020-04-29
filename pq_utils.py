import sys
import os
import pandas as pd
import numpy as np
from model import model as mdl
from scipy.stats.stats import pearsonr
from keras import backend as K
import pickle
import argparse
from utilsGGG import *
from keras.models import load_model

def prep_data(data,args,p=None,loops=None):

    flank = args['flank']

    if args['model_feat'] == 'base':
        if p:
            max_seq = p
        else:
            max_seq = data.apply(len).max()

        X_oh = np.stack(data.apply(lambda x: oneHot(x, p)).values)
        X_in = [X_oh]

    else:
        X = data.str[flank:-flank]

        if p:
            max_seq = p
        else:
            max_seq = X.apply(len).max()

        f1 = data.str[:flank]
        f2 = data.str[-flank:]

        X_oh = np.stack(X.apply(lambda x: oneHot(x, max_seq)).values)
        f1_oh = np.stack(f1.apply(lambda x: oneHot(x)).values)
        f2_oh = np.stack(f2.apply(lambda x: oneHot(x)).values)

        X_in = [X_oh, f1_oh, f2_oh]

        if args['model_feat'] == 'split_numloops':
            loop_in = loops.apply(len)
            loop_in = np.expand_dims(loop_in.to_numpy(), 1)
            X_in.append(loop_in)

        elif args['model_feat'] == 'split_looplen':
            max_loops = max(loops.apply(len))
            loop_in = np.stack(loops.apply(
                lambda x: np.pad(x, (0, max_loops - len(x)), 'constant', constant_values=0)).values)
            X_in.append(loop_in)

    return X_in


