from numpy.lib.function_base import iterable
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

# sns.set(style="whitegrid")

df = pd.read_csv('houses_data.csv')
df_new = df.loc[:, ('Distance', 'Car', 'Bathroom')]


# #
# df_new.boxplot(return_type='dict')
# plt.plot()
# plt.show()
# boxplot = df.boxplot(column=['Car', 'Rooms', 'Bathroom'])
# #


# print(df_new.describe())
print(df_new.dtypes)
def get_quantivative_date(df: pd.DataFrame):
    return df.select_dtypes(include=np.number)


anomalies = []


def find_anomalies(data):
    std = np.std(data)
    mean = np.mean(data)
    anomaly_cut_off = std * 3

    lower_limit = mean - anomaly_cut_off
    upper_limit = mean + anomaly_cut_off
    print(lower_limit)
    print(upper_limit)
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


find_anomalies(dff)
dff
df['col'] = df['col'].astype('str')
def generate_statistics(df, estimator: list = None, map_estimator_names: dict = None):
    df_statistics = df.describe()

    if estimator:
        df_statistics = df_statistics.loc[estimator]

    if map_estimator_names:
        df_statistics = df_statistics.rename(index=map_estimator_names)

    return df_statistics.T


def outliners(df: pd.DataFrame):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    bool_filter = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
    return bool_filter


def remove_outliners(df: pd.DataFrame, just_numerics=False):
    if just_numerics:
        df = get_quantivative_date(df)
    bool_filter = outliners(df)
    return df.mask(bool_filter)


def fill_with_median(df: pd.DataFrame):
    return df.fillna(df.median())


# print(get_quantivative_date(df=df_new))
# print(generate_statistics(df=df_new))
# print(outliners(df=df_new))
# print(remove_outliners(df=df_new))
# print(fill_with_median(df=df_new))

#
# df_new.columns.tolist()
# print(df_new.columns.tolist())
# Q2 = df_new.quantile(0.50) #median
# Q1 = df_new.quantile(0.25)
# Q3 = df_new.quantile(0.75)
#
# IQR = (Q3 - Q1) * 1.5
# batas_bawah = Q1 - IQR
# batas_atas = Q3 + IQR
#
# def find_outlier(data):
#     outlier = ''
#     for i in range(len(data)):
#         if data[i] > batas_atas:
#             outlier = outlier + str(data[i]) + ', '
#         if data[i] < batas_bawah:
#             outlier = outlier + str(data[i]) + ', '
#     return outlier
#
# find_outlier(df_new)
# print(df_new < (Q1 - 1.5 * IQR)) |(df_new > (Q3 + 1.5 * IQR))


#
# Q1 = col.quantile(0.25)
# Q3 = col.quantile(0.75)
# IQR = Q3 - Q1

# print(df < (Q1 - 1.5 * IQR))
# print(df > (Q3 + 1.5 * IQR))

# df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(df_out.shape)
# print(IQR)

# 1. Napisz n funkcji do wykrywania outierów
# 2. wybierz 3 kolumny
# 3. korzystając z for loop wyświetl liczbę wartości odstające dla każdej z metod
# 4. utwórz n-nowych DF bez outlierów
# 5. Dla każdego nowego DF'a policz MAE
