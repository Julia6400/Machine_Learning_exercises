# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import median_abs_deviation, zscore

# 1. ZADANIE WSTĘPNE

# Dla 3 kolumn o numerycznych wartościach przedstaw:


df = pd.read_csv('houses_data.csv', parse_dates=['Date'])
df.drop('Date', axis=1, inplace=True)
df_new = df.loc[:, ('Distance', 'Car', 'Bathroom')]

print(df_new.dtypes)
attributes = list(df_new.head())

# boxplot
sns.boxplot(data=df_new, orient="h", palette="Set2").set_title("Outliners")
plt.show()


# wyznacz Q1, Q3, IQR oraz SD (standard dev.) + średnia i mediana

def get_quantivative_date(df: pd.DataFrame):
    return df.select_dtypes(include=np.number)


def generate_statistics(df, estimators: list = None, map_names_estimators: dict = None):
    df_statistics = df.describe()
    if estimators:
        df_statistics = df_statistics.loc[estimators]
    if map_names_estimators:
        df_statistics = df_statistics.rename(index=map_names_estimators)
    return df_statistics.T


def mean_statistics(df: pd.DataFrame):
    data = {"Attributes": attributes,
            "Mean": df.mean().tolist(),
            "Median": df.median().tolist(),
            "Std": df.std().tolist(),
            "Q1": df.quantile(0.25).tolist(),
            "Q3": df.quantile(0.75).tolist(),
            "IQR": (df.quantile(0.25) - df.quantile(0.75)).tolist()}
    frame = pd.DataFrame(data)
    frame = frame.set_index(["Attributes"])
    return frame


# print(get_quantivative_date(df_new))
# print(generate_statistics(df_new))
# print(mean_statistics(df_new))


# 2. ZADANIE GŁÓWNE


# log transform
def log_transform(df: pd.DataFrame):
    df = df.replace(0, np.nan)
    return np.log(df)


# usuwanie za pomocą IQR

def IQR(df: pd.DataFrame):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    bool_filter = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
    return bool_filter


# Uzupełnianie za pomocą mediany

def fill_with_median(df: pd.DataFrame):
    fillna = df.ffill().bfill().astype(int)
    return fillna


# removing 0.1 & 0.9 percentile

def percentile_method(df: pd.DataFrame):
    percentile = df < np.percentile(df, 90)
    percentile = df > np.percentile(df, 10)
    return percentile


# 3sigma
def sigma_oulier(df: pd.DataFrame):
    column_mean = df.mean()
    sigma = 3 * df.std(ddof=0)
    lower_limit = column_mean - sigma
    upper_limit = column_mean + sigma
    return (df < lower_limit) | (df > upper_limit)


# zscore

def zscore_outlier(df: pd.DataFrame):
    return zscore(df) > 2


# z score outlier

def modified_z_score_outlier(df: pd.DataFrame):
    mad_column = median_abs_deviation(df)
    median = np.median(df)
    mad_score = np.abs(0.6745 * (df - median) / mad_column)
    return mad_score > 3.5


outliers_methods_dict = {
    "log transform": log_transform,
    "3sigma": sigma_oulier,
    "z score": zscore_outlier,
    "mod z score": modified_z_score_outlier,
    "fill with median": fill_with_median,
    "percentile method (0.1-0.9)": percentile_method,
    "IQR": IQR
}

for method_name, method in outliers_methods_dict.items():
    print("\nRunning method: ", method_name)
    print(df_new.apply(lambda x: method(x)).sum())


# usuwanie outlierów za pomocą IQR

def remove_outliners(df: pd.DataFrame, just_numerics=False):
    if just_numerics:
        df = get_quantivative_date(df)
    bool_filter = IQR(df)
    return df.mask(bool_filter)


#print(remove_outliners(df_new))
