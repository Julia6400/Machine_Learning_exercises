# libraries

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# 1. ZADANIA GŁÓWNE

# Def function where only specific columns will be loaded

def load_file(path: str, cols: List[str] = None) -> pd.DataFrame:
    if cols is not None:
        df = pd.read_csv(path, usecols=cols)
        return df.select_dtypes(include=['number'])

    else:
        df = pd.read_csv(path, usecols=cols)
        return df.select_dtypes(include=['number'])


# path
df = load_file('houses_data.csv')

print("Data types:\n", df.dtypes)
print("\n Data frame: \n", df)


# replacing null values methods
def methods():
    # Simple imputer
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    mean_imputer = mean_imputer.fit(df[["Car", "Rooms", "Price"]])
    imputed_df = mean_imputer.transform(df[["Car", "Rooms", "Price"]])
    print("\n Simple imputer: \n", imputed_df)

    # drop axis 1 - columns, 0 - rows
    drop_axis_1 = df.dropna(axis=1)
    drop_axis_0 = df.dropna(axis=0)
    print("\n Drop axis 1: \n", drop_axis_1)
    print("\n Drop axis 0: \n", drop_axis_0)

    # drop all null rows
    drop_all = df.dropna(how='all')
    print("\n Drop all null rows: \n", drop_all)

    # drop tresh - keep rows with at least 3 non null vals
    drop_thresh = df.dropna(thresh=3)
    print("\n Drop tresh: \n", drop_thresh)

    # drop col - drop all nulls values from specific column/s
    drop_col = df.dropna(subset=['Rooms'])
    print("\n Drop column: \n", drop_col)

    # non nans
    df_non_nans = pd.DataFrame(df, columns=[["Car", "Rooms", "Price"]])
    print("\n Non nans: \n", df_non_nans)

    # drop NAN
    df_dropna_method = pd.DataFrame.dropna(df)
    print("\n Drop Nan: \n", df_dropna_method)

    # fill NAN
    df_fillna_mean = df.fillna(df.mean())
    print("\n Fill NaN mean: \n", df_fillna_mean)

    # KNNImputer
    impt = KNNImputer()
    df[['BuildingArea', 'YearBuilt', 'Car']] = impt.fit_transform(df[['BuildingArea', 'YearBuilt', 'Car']])

    return print(" ^                                                 ^\n "
                 "|                                                 "
                 "|\n | Some of basic methods to deal will null values. |\n")


methods()

# MAE
houses_predictors = df[['Car', 'BuildingArea', 'Distance', 'Bathroom']]
houses_target = df['Price']

X_train, X_test, y_train, y_test = train_test_split(houses_predictors, houses_target,
                                                    train_size=0.7, test_size=0.3, random_state=0)


def score_dataset(X_train, X_test, y_train, y_test):
    regr_model = LinearRegression()
    regr_model.fit(X_train, y_train)
    preds = regr_model.predict(X_test)

    regressor_model = RandomForestRegressor()
    regressor_model.fit(X_train, y_train)
    preds_regressor = regressor_model.predict(X_test)

    return mean_absolute_error(y_test, preds), mean_absolute_error(y_test, preds_regressor)


result = score_dataset(X_train, X_test, y_train, y_test)
print("\n Regression MAE: \n", result)
