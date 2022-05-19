from sklearn import linear_model, datasets, metrics, model_selection
# from scipy import stats
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

diabetes = datasets.load_diabetes()
# x = diabetes.data[:, np.newaxis, 2]
# y = diabetes.target

X, y = diabetes.data, diabetes.target
lin_regr_model = LinearRegression()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
lin_regr_model.fit(X_train, y_train)
y_pred = lin_regr_model.predict(X_test)

alpha_space = np.logspace(-4, 0, 50)
lasso = Lasso()
ridge = Ridge()

for alpha in alpha_space:
    ridge.set_params(alpha=alpha)

print("Prediction:")
print(y_pred)

print("RMSE:")
print(sqrt(mean_squared_error(y_pred, y_test)))

print("R2:")
print(r2_score(y_pred, y_test))

print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))

#
#
# plt.scatter(x_test, y_test, color='black')
# plt.plot(x_test, linearRegressionModel.predict(x_test), color='blue',
#          linewidth=2)
# plt.xticks()
# plt.yticks()
# plt.show()


alpha_space = np.logspace(-4, 0, 50)
ridge_score = []
