import numpy as np
import pandas as pd
from datetime import *
import os


df = pd.read_csv('iono.data', header=None)
cols = ['f{0}'.format(i) for i in xrange(34)] + ['class']
y = map(lambda o: 'c1' if o == 'b' else 'c0', df.ix[:, 34])
X = np.array(df.ix[:, :33])
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
df = pd.DataFrame(np.hstack((X, zip(y))))
df.columns = cols
df=df.drop('f1', axis=1)
df.to_csv('iono-norm.data', index=False)
