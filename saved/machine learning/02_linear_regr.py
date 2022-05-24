from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import pandas as pd


diabetes = load_diabetes()
dir(diabetes)
X, y = diabetes.data, diabetes.target

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X, y)
preds = dummy_regr.predict(X)
print("RMSE for dummy regression", mean_squared_error(y, preds, squared=False))
print("r2 score for dummy regression", r2_score(y, preds))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_std, y_train)
preds = model.predict(X_test_std)

print("RMSE for linear regression", mean_squared_error(y_test, preds, squared=False))
print("r2 score for linear regression", r2_score(y_test, preds))

clf = Ridge()
clf.fit(X_train_std, y_train)
preds = clf.predict(X_test_std)
print("RMSE for ridge", mean_squared_error(y_test, preds, squared=False))
print("r2 score for ridge", r2_score(y_test, preds))

clf = Lasso()
clf.fit(X_train_std, y_train)
preds = clf.predict(X_test_std)
print("RMSE for lasso", mean_squared_error(y_test, preds, squared=False))
print("r2 score for lasoo", r2_score(y_test, preds))

alpha = list(range(1, 101, 1))
score_list_ridge = []
score_list_lasso = []

for i in alpha:
    ridge = Ridge()
    ridge.set_params(alpha=i)
    ridge.fit(X_train_std, y_train)

    preds = ridge.predict(X_test_std)
    score = r2_score(y_test, preds)

    score_list_ridge.append(score)

for i in alpha:
    lasso = Lasso()
    lasso.set_params(alpha=i)
    lasso.fit(X_train_std, y_train)

    preds = lasso.predict(X_test_std)
    score = r2_score(y_test, preds)

    score_list_lasso.append(score)
plt.figure(figsize=(10, 5))
sns.lineplot(x=alpha, y=score_list_ridge, label="ridge")
sns.lineplot(x=alpha, y=score_list_lasso, label="lasso")
plt.title("R2 score for alpha range")
plt.legend()
plt.grid()
plt.show()

alpha = (0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000)
cv = list(range(2, 42, 2))
score_list_ridge = []
score_list_lasso = []

for i in cv:
    ridge = RidgeCV(alphas=alpha, cv=i)
    ridge.fit(X_train_std, y_train)
    preds = ridge.predict(X_test_std)

    score = r2_score(y_test, preds)
    score_list_ridge.append(score)
print(score)

for i in cv:
    lasso = LassoCV(alphas=alpha, cv=i)
    lasso.fit(X_train_std, y_train)
    preds = lasso.predict(X_test_std)

    score = r2_score(y_test, preds)
    score_list_lasso.append(score)
print(score)

plt.figure(figsize=(10,5))
sns.lineplot(x=cv, y=score_list_ridge, label="ridgeCV")
sns.lineplot(x=cv, y=score_list_lasso, label="lassoCV")
plt.xticks(range(2, 42, 2))
plt.title("R2 score for CV range")
plt.legend()
plt.grid()
plt.show()

r2_score_dict = {
    "dummy regression": 0.0,
    "linear regression": 0.4033025232246107,
    "ridge": 0.4027277632830567,
    "lasoo": 0.40050373260020367,
    "ridgeCV": 0.4045745545779539,
    "lassoCV": 0.40050373260020367
}
r2_df = pd.DataFrame.from_dict(r2_score_dict, orient='index')
r2_df

plt.figure(figsize=(10,5))
sns.barplot(x=r2_df.index, y=r2_df[0])
plt.ylim(0.40, 0.41)
plt.grid()
plt.show()
