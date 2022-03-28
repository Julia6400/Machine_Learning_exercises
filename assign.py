# libraries

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# 1. ZADANIA WSTĘPNE

# read data
df = pd.read_csv('houses_data.csv')

# checking and deleting (if are) any duplicates
df.drop_duplicates()

# checking data types
print(df.dtypes)

# columns with missing values
count_nan = df.isnull().sum().sum()
count_nan_in_df = df.isnull().sum()
print('Count of NaN: ' + str(count_nan))
print(count_nan_in_df)

missing_values = df.isnull().sum() / len(df)
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
print(missing_values)

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
print(diff.sort_values(by=['Corelation vs col price']))


# checking variable number of rooms
df_count = df.groupby('Rooms')['Price'] \
       .agg(count_rooms='size', mean_price='mean') \
       .reset_index().round(2)
print(df_count)

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


