# libraries
import pandas as pd
from matplotlib import pyplot as plt
import csv
import os


# table and dataframes
def main():
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    data = [x, y1, y2, y3, x4, y4]
    anscombe_df = pd.DataFrame(data).transpose()
    anscombe_df.columns = ['x', 'y1', 'y2', 'y3', 'x4', 'y4']
    anscombe_y_values = anscombe_df.iloc[:, lambda anscombe_df: [1, 2, 3, 5]]

# equations for mean, standard deviation, variation and pearson
    anscombe_equations = pd.DataFrame()
    anscombe_equations['mean'] = anscombe_y_values.mean().round(2)
    anscombe_equations['standard deviation'] = anscombe_y_values.std().round(2)
    anscombe_equations['variation'] = anscombe_y_values.var().round(2)
    anscombe_equations = anscombe_y_values.corr(method='pearson')

# creating new folder and adding csv file

    os.mkdir('results_folder')
    anscombe_equations.to_csv(os.path.join('results_folder', 'result.csv'))

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

#plot save
    plt.savefig('results_folder/anscombe_plot.jpg')


if __name__ == '__main__':
    main()

