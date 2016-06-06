import numpy as np
import pandas as pd
from datetime import *
import os


df = pd.read_csv('haber.data', header=None)
cols = ['f{0}'.format(i) for i in xrange(3)] + ['class']
y = map(lambda o: 'c1' if o == 2 else 'c0', df.ix[:, 3])
X = np.array(df.ix[:, :2]).astype('float')
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
df = pd.DataFrame(np.hstack((X, zip(y))))
df.columns = cols
df.to_csv('haber-norm.data', index=False)
