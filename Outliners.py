import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import median_abs_deviation, zscore


# 1. ZADANIE WSTĘPNE

# 3 kolumny
df = pd.read_csv('houses_data.csv')
df_new = df.loc[:, ('Distance', 'Car', 'Bathroom')]

print(df_new.dtypes)
attributes = list(df_new.head())

# boxplot
sns.boxplot(data=df_new, orient="h", palette="Set2")
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
            "SD": df.std().tolist(),
            "Q1": df.quantile(0.25).tolist(),
            "Q3": df.quantile(0.75).tolist(),
            "IQR": (df.quantile(0.25) - df.quantile(0.75)).tolist()}
    DataFrame = pd.DataFrame(data)
    DataFrame = DataFrame.set_index(["Attributes"])
    return DataFrame


# 2. ZADANIE GŁÓWNE

# log transform
def log_transform_method(df: pd.DataFrame):
    columns = df.columns
    for col in columns:
        df[col] = np.log(df[col])
    return columns


# usuwanie za pomocą IQR

def IQR(df: pd.DataFrame):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    bool_filter = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)
    return bool_filter


# Uzupełnianie za pomocą mediany

def fill_with_median(df: pd.DataFrame):
    return df.fillna(df.median())


# removing 0.1 & 0.9 percentile

def percentile_method(df: pd.DataFrame):
    percentile_1 = df < np.percentile[df, 0.90],
    percentile_2 = df > np.percentile[df, 0.10]
    return percentile_1, percentile_2

# zscore

def zscore_outlier(df: pd.DataFrame):
    return zscore(df) > 2


def modified_z_score_outlier(df: pd.DataFrame):
    mad_column = median_abs_deviation(df)
    median = np.median(df)
    mad_score = np.abs(0.6745 * (df - median) / mad_column)
    return mad_score > 3.5


outliers_methods_dict = {
    "log transform": log_transform_method,
    "z_score": zscore_outlier,
    "fill_with_median": fill_with_median,
    "per_metod_0.1-0.9": percentile_method,
    "IQR": IQR,
    "mod_z_score": modified_z_score_outlier
}

for method_name, method in outliers_methods_dict.items():
    print("\nRunning method: ", method_name)
    print(df_new.apply(lambda x: method(x)).sum())


def remove_outliners(df: pd.DataFrame, just_numerics=False):
    if just_numerics:
        df = get_quantivative_date(df)
    bool_filter = IQR(df)
    return df.mask(bool_filter)


# MAE
# houses_predictors = df[['Distance', 'Car', 'Bathroom']]
# houses_target = df['Price']
#
# X_train, X_test, y_train, y_test = train_test_split(houses_predictors, houses_target,
#                                                     train_size=0.7, test_size=0.3, random_state=0)
#
#
# def score_dataset(X_train, X_test, y_train, y_test):
#     regr_model = LinearRegression()
#     regr_model.fit(X_train, y_train)
#     preds = regr_model.predict(X_test)
#
#     regressor_model = RandomForestRegressor()
#     regressor_model.fit(X_train, y_train)
#     preds_regressor = regressor_model.predict(X_test)
#
#     return mean_absolute_error(y_test, preds), mean_absolute_error(y_test, preds_regressor)
#
#
# result = score_dataset(X_train, X_test, y_train, y_test)
# print("\n Regression MAE: \n", result)
#
#
# def score_dataset(X_train, X_test, y_train, y_test):
#     regr_model = LinearRegression()
#     regr_model.fit(X_train, y_train)
#     preds = regr_model.predict(X_test)
#
#     return mean_absolute_error(y_test, preds)
#
#

#
# #print(get_quantivative_date(df=df_new))
# print(generate_statistics(df=df_new))
# print(mean_statistics(df=df_new))
# # print(remove_outliners(df=df_new))
# # print(fill_with_median(df=df_new))
