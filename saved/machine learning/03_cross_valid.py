
import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

wine = load_wine()
lin_regr_model = LinearRegression()
X, y = wine.data, wine.target

splits = 2

kf = KFold(n_splits=splits)
kf.get_n_splits(X)

skf = StratifiedKFold(n_splits=splits)
skf.get_n_splits(X, y)

# X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

#
# y_pred = knn.predict(X_test)


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    scores = cross_val_score(dt, data_train[cols], data_train.quality, cv=10, scoring='accuracy')


for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



# print(len(X) // splits)
#
# for fold_nb, (train_index, test_index) in enumerate(kf.split(X)):
#     print("Fold nb: ", fold_nb)
#     print("Train size: ", len(train_index), "Test size: ", len(test_index))


from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
# X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
# skf = StratifiedKFold(n_splits=3)

# print("Method Stratifield-K-Fold :\n")
# for train, test in skf.split(X, y):
#     print('train -  {}   |   test -  {}'.format(
#         np.bincount(y[train]), np.bincount(y[test])))
# print("\nMethod K fold :\n")
# kf = KFold(n_splits=3)
# for train, test in kf.split(X, y):
