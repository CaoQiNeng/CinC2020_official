import pandas as pd
import numpy as np
from common import *

DATA_DIR = DATA_ROOT_PATH + '/training2017'
df = pd.read_csv(DATA_DIR + '/REFERENCE_V2.csv') #.fillna('')

ids = df['ids'].values

a1 = np.random.choice(a = int(len(ids)), size=3, replace=False, p=None)