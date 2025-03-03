import numpy as np
import pandas as pd

np.random.seed(42)
X = np.linspace(0, 10, 1000000)
Y = 3 * X + 5 + np.random.normal(0, 2, 1000000)

pd.DataFrame({'X': X, 'Y': Y}).to_csv('test_data.csv', index=False)