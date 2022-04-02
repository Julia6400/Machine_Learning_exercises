# libraries

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. ZADANIA WSTĘPNE

# read data
df = pd.read_csv('houses_data.csv')

# checking and deleting (if are) any duplicates
df.drop_duplicates()

# checking data types
print("\n Data types: \n", df.dtypes)

# columns with missing values
count_nan = df.isnull().sum().sum()
count_nan_in_df = df.isnull().sum()
# print('\nTotal count of NaN: \n' + str(count_nan))
# print('\nCount of NaN in specific columns: \n', count_nan_in_df)

missing_values = df.isnull().sum() / len(df)
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
# print("\n Missing values:\n", round(missing_values,3))

missing_values = missing_values.to_frame()
missing_values.columns = ['count']
missing_values.index.names = ['Column name']
missing_values['Column name'] = missing_values.index


# barplot for missing values
ax = sns.barplot(x='Column name', y='count', data=missing_values, hue='Column name')
sns.set(style="whitegrid", color_codes=True)
plt.xticks(rotation=90)
ax.set_title('Missing values')
ax.set_ylabel("Percent")
ax.set_xlabel("Column name")
ax.yaxis.set_major_formatter(PercentFormatter(1))


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)


change_width(ax, 0.80)

plt.show()


# correlation for numbers columns vs column "PRICE"
df_corr = df.select_dtypes(['float64', 'int64'])
diff = []

for col in df_corr:
    if col == "Price":
        pass
    else:
        a = df[col].corr(df['Price'])
        diff += [[col, a]]

diff = pd.DataFrame(diff, columns=['Column', 'Corelation vs col price'])
print("\n Correlation: \n", diff.sort_values(by=['Corelation vs col price']))


# checking variable number of rooms
df_count = df.groupby('Rooms')['Price'] \
       .agg(count_rooms='size', mean_price='mean') \
       .reset_index().round(2)
print("\n Number of rooms in the apartments: \n", df_count)

# choosing interval for rooms
rooms_2 = df.loc[df['Rooms'] == 2]
rooms_3 = df.loc[df['Rooms'] == 3]
rooms_4 = df.loc[df['Rooms'] == 4]

bins = 50

# histogram for number of rooms vs price
sns.set_color_codes("deep")

rel = sns.histplot(rooms_3, x="Price", bins=bins, color="y")
rel = sns.histplot(rooms_4, x="Price", bins=bins, color="b")
rel = sns.histplot(rooms_2, x="Price", bins=bins, color="m")
rel.set_title('Apartments with 2 to 4 rooms vs theirs prices')
rel.set_ylabel('Rooms')
rel.legend(labels=["Apartments with 3 rooms", "Apartments with 4 rooms","Apartments with 2 rooms"], title = "Legend")
plt.show()


# 2. ZADANIA GŁÓWNE

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