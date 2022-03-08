# libraries
import numpy as np
import pandas as pd
import altair as alt
import altair_viewer
import seaborn
import seaborn as sns

# dataset
data = seaborn.load_dataset('penguins')
degree_list = [1, 3, 5]

# visualization
base = alt.Chart(data).mark_circle(color="black").encode(alt.X("bill_length_mm"), alt.Y("flipper_length_mm"))

polynomial_fit = [base.transform_regression(
    "bill_length_mm", "flipper_length_mm", method="poly", order=order, as_=["bill_length_mm", str(order)])
    .mark_line()
    .transform_fold([str(order)], as_=["degree", "flipper_length_mm"])
    .encode(alt.Color("degree:N"))
    for order in degree_list]

# html save

alt.layer(base, *polynomial_fit)\
    .encode(tooltip=['bill_length_mm', 'flipper_length_mm'])\
    .interactive()\
    .save('chart.html')\
    .show()
