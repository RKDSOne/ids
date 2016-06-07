import numpy as np
import pandas as pd
from datetime import *
import os


def more_imbalanced(fname, sample_rate):
    df = pd.read_csv(fname)
    print df['class'].value_counts()
    cols = df.columns
    ar = np.array(df)
    idx = (ar[:, -1] == 'c0') | (np.random.rand(ar.shape[0]) <= sample_rate)
    df = pd.DataFrame(ar[idx])
    df.columns = cols
    print df['class'].value_counts()
    df.to_csv('sampled' + fname, index=False)
