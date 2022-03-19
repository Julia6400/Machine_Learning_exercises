import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('houses_data.csv')

count_nan = df.isnull().sum().sum()
count_nan_in_df = df.isnull().sum()
df.drop_duplicates()
print(df.dtypes)
print('Count of NaN: ' + str(count_nan))
print(count_nan_in_df)

missing_values = df.isnull().sum() / len(df)*100
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
print(missing_values)

missing_values = missing_values.to_frame()
missing_values.columns = ['count']
missing_values.index.names = ['Name']
missing_values['Name'] = missing_values.index


sns.set(style="whitegrid", color_codes=True)
sns.barplot(x='Name', y='count', data=missing_values)
plt.xticks(rotation=90)
plt.show()


