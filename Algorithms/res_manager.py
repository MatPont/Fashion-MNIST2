import sys
import pandas as pd
from statistics import mean, stdev

df = pd.read_csv(sys.argv[1])
print(mean(df.iloc[:,0]))
print(stdev(df.iloc[:,0]))
print(mean(df.iloc[:,1]))
print(stdev(df.iloc[:,1]))
