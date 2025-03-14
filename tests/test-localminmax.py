# import necessary libraries
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# create a sample series
s = pd.Series([1, 3, 2, 4, 3, 5, 4, 6, 5, 4])

# use shift() function
local_max_indices = argrelextrema(s.values, np.greater)
local_max = s[local_max_indices[0]]

# print the results
print(local_max)
