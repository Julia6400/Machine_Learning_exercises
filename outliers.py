import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

df = pd.read_csv('houses_data.csv')

boxplot = df.boxplot(column=['Car', 'Rooms', 'Bathroom'])

df_new = df[['Car', 'Rooms', 'Bathroom']]
print(df_new.describe())
print(df_new.dtypes)

# Q1 = df_new.quantile(0.25)
# Q3 = df_new.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# print(df_new < (Q1 - 1.5 * IQR)) |(df_new > (Q3 + 1.5 * IQR))
# plt.show()


col = df[['Car', 'Rooms', 'Bathroom']]
print(col)

Q1 = col.quantile(0.25)
Q3 = col.quantile(0.75)
IQR = Q3 - Q1

# print(df < (Q1 - 1.5 * IQR))
# print(df > (Q3 + 1.5 * IQR))

df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_out.shape)
print(IQR)

# 1. Napisz n funkcji do wykrywania outierów
# 2. wybierz 3 kolumny
# 3. korzystając z for loop wyświetl liczbę wartości odstające dla każdej z metod
# 4. utwórz n-nowych DF bez outlierów
# 5. Dla każdego nowego DF'a policz MAE
