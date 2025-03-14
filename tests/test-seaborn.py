import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns

#df = sm.datasets.co2.load(as_pandas=True).data
df = sm.datasets.co2.load().data
df['month'] = pd.to_datetime(df.index).month
df['year'] = pd.to_datetime(df.index).year
sns.lineplot(x='month',y='co2',hue='year',data=df.query('year>1995')) 
plt.show()

