# libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv


# table

def main():
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]


    data = [x, y1, y2, y3, x4, y4]

    df = pd.DataFrame(data).transpose()

    df.columns = ['x', 'y1', 'y2', 'y3', 'x4', 'y4']

    df_y = df.iloc[:, lambda df : [1, 2, 3, 5]]


    df_disc = {
        'I': y1,
        'II': y2,
        'III': y3,
        'IV': y4,
    }

    anscombe_df = pd.DataFrame(df_disc, index=x)
    anscombe_df.plot(subplots=True, figsize=(5, 15), style="o", ms=10)

    anscombe_df = pd.DataFrame()
    anscombe_df['mean'] = df_disc.mean().round(2)
    anscombe_df['std'] = df_disc.std().round(2)
    anscombe_df['var'] = df_disc.var().round(2)
    anscombe_df_pearson = df_disc.corr(method='pearson')


    with open('result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(anscombe_df)

    fig, axs = plt.subplots(2, 2)

    # I plot
    axs[0, 0].scatter(x, y1, c='red')
    axs[0, 0].set_title("I", loc='center', fontweight='bold')
    axs[0, 0].set_ylim(0, 14)
    axs[0, 0].set_xlim(0, 20)

    # II plot
    axs[1, 0].scatter(x, y2, c='blue')
    axs[0, 1].set_title("II", loc='center', fontweight='bold')
    axs[1, 0].set_ylim(0, 14)
    axs[1, 0].set_xlim(0, 20)

    # III plot
    axs[0, 1].scatter(x, y3, c='green')
    axs[1, 0].set_title("III", loc='center', fontweight='bold')
    axs[0, 1].set_ylim(0, 14)
    axs[0, 1].set_xlim(0, 20)

    # IV plot
    axs[1, 1].scatter(x4, y4, c='purple')
    axs[1, 1].set_title("IV", loc='center', fontweight='bold')
    axs[1, 1].set_ylim(0, 14)
    axs[1, 1].set_xlim(0, 20)

    fig.set_figheight(8)
    fig.set_figwidth(8)
    plt.show()


if __name__ == '__main__':
    main()
# anscombe_df.mean().round(2)
# print(anscombe_df.mean().round(2))

