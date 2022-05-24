import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LassoCV, Lasso

diabetes = load_diabetes()
dir(diabetes)

X, y = diabetes.data, diabetes.target
kf = KFold(n_splits=3, shuffle=True, random_state=1)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("TRAIN:", train_index, "\n" "TEST:", test_index)


def score_by_fold():
    alphas = np.logspace(-4, -0.5, 30)
    lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
    k_fold = KFold(10)

    mean_list = []

    for k, (train, test) in enumerate(k_fold.split(X, y)):
        lasso_cv.fit(X[train], y[train])

        score = lasso_cv.score(X[test], y[test])
        mean_list.append(score)
        print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
              format(k, lasso_cv.alpha_, score))

    print("Mean score: ", np.mean(mean_list))


print(score_by_fold())


def my_cross_val():
    list = [3, 5, 10]
    lasso = Lasso()

    for n in list:
        print(cross_val_score(lasso, X, y, cv=n))
        print("Max: ", max(cross_val_score(lasso, X, y, cv=n)), "\n")


print(my_cross_val())

r2_score_dict = {
    "dummy regression": 0.0,
    "linear regression": 0.4033025232246107,
    "ridge": 0.4027277632830567,
    "lasoo": 0.40050373260020367,
    "ridgeCV": 0.4045745545779539,
    "lassoCV": 0.40050373260020367,
    # scores from 03_cross_valid
    "lasso_cv_kfold_10": 0.6827010716027995,
    "lasso_cross_val_10": 0.4287427630907267
}
r2_df = pd.DataFrame.from_dict(r2_score_dict, orient='index', columns=["value"])
sorted_r2_df = r2_df.sort_values(by=["value"])
sorted_r2_df

plt.figure(figsize=(10, 5))
sns.barplot(x=r2_df.index, y=r2_df.value)
plt.ylim(0.3, 0.7)
plt.xticks(rotation=45)
plt.title("R2 scores for regression models")
plt.grid()
plt.show()
