# libraries
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os

# create dir to save newly emerging data and plot
SAVE_DIR = os.path.join(os.getcwd(), "../saved")


# initialize Anscombe's quartet data and create DataFrame
def anscombe_data():

    """
    This function loads the Anscombe's quartet from the Seaborn's library resources
    and calculates mean, standard deviation and variance
    Args:
        None
    Returns:
          df - dataframe with Anscombe's x and y values
          description - statistical description of Anscombe's datasets
    """

    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    df_dict = {"y1": y1, "y2": y2, "y3": y3, "y4": y4}
    anscombe_df = pd.DataFrame(df_dict, index=x)

    return anscombe_df, x, x4


def stats(data):
    """
    data frame with calculated mean and standard deviation
    Calculating variance and merging both dataframes
    """

    description = data.describe().loc[["mean", "std"]]
    variance = pd.DataFrame({"var": data.var()})
    description = description .T.merge(variance, left_index=True, right_index=True)

    return description


# scatter plot of Anscombe's quartet data
def anscombe_plot(anscombe_df, x, x4):

    """
    This function creates chart for each dataset
    Args:
        data: pd.DataFrame
    Returns:
         charts: subplot of 4 graphs
    """

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 5))
    axs[0, 0].set(xlim=(0, 20), ylim=(2, 14))
    axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12))
    titles = ["I", "II", "III", "IV"]
    x_labels = ["x", "x", "x", "x4"]
    y_labels = ["y1", "y2", "y3", "y4"]
    x_i = [x, x, x, x4]
    y_i = [anscombe_df["y1"], anscombe_df["y2"], anscombe_df["y3"], anscombe_df["y4"]]
    k = 0
    for i in range(2):
        for j in range(2):
            axs[i, j].scatter(x_i[k], y_i[k])
            axs[i, j].set_title(titles[k])
            axs[i, j].set_xlabel(x_labels[k])
            axs[i, j].set_ylabel(y_labels[k])
            k += 1
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass
    anscombe_df, x, x4 = anscombe_data()
    description = stats(anscombe_df)
    description.to_csv(os.path.join(SAVE_DIR, "stats.csv"))
    fig = anscombe_plot(anscombe_df, x, x4)
    fig.savefig(os.path.join(SAVE_DIR, "chart.jpg"))


print(anscombe_data())
print(stats(anscombe_df))
matplotlib.pyplot.show()
