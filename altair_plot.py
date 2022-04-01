# libraries
import altair as alt
import seaborn

# data load
data = seaborn.load_dataset('penguins')

# Define the degree of the polynomial fits
degree_list = [1, 3, 5]

base = alt.Chart(data).mark_circle(color="black").encode(
    alt.X("bill_length_mm"), alt.Y("flipper_length_mm"))

base = alt.Chart(data).mark_circle(color="black").encode(
    alt.X("bill_length_mm", scale=alt.Scale(zero=False)),
    alt.Y("flipper_length_mm", scale=alt.Scale(zero=False)),
    tooltip="flipper_length_mm")

"""
Polynomial Fit Plot with Regression Transform
Multiple fitted polynomials using the regression transform.
"""

polynomial_fit = [
    base.transform_regression(
        "bill_length_mm", "flipper_length_mm", method="poly", order=order, as_=["bill_length_mm", str(order)])
        .mark_line()
        .transform_fold([str(order)], as_=["degree", "flipper_length_mm"])
        .encode(alt.Color("degree:N"))
    for order in degree_list]

# making it interactive and save to html
alt.layer(base, *polynomial_fit) \
    .encode(tooltip=['bill_length_mm', 'flipper_length_mm']) \
    .interactive() \
    .show()

base.save('altair.html')
