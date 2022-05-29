# Libraries
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


# Data set
wine = load_wine()
features = wine.data
target = wine.target

logistic = LogisticRegression(solver="liblinear", max_iter=500)
penalty = ["l1", "l2"]

C = np.logspace(0, 4, 1000)
hyperparameters = dict(C=C, penalty=penalty)

# Grid search
GridSearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=2, n_jobs=-1)
best_model = GridSearch.fit(features, target)

# Randomized Search
RandomizedSearch = RandomizedSearchCV(logistic, hyperparameters, n_iter=100, cv=5, verbose=0, n_jobs=-1)
best_randomized_model = RandomizedSearch.fit(features, target)

print("Best Penalty:", best_model.best_estimator_.get_params()['penalty'])
print("Best C:", best_model.best_estimator_.get_params()['C'])

print("Best Penalty in random model:", best_randomized_model.best_estimator_.get_params()['penalty'])
print("Best C in random model:", best_randomized_model.best_estimator_.get_params()['C'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logistic_2 = LogisticRegression(solver="liblinear", max_iter=200)
pipe = Pipeline([("classifier", RandomForestClassifier())])

search_space = [{"classifier": [logistic_2],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]

GridSearch_pipe = GridSearchCV(pipe, search_space, cv=5, verbose=1, n_jobs=-1)
best_model_pipe = GridSearch_pipe.fit(features, target)

print("Best classifier:", best_model_pipe.best_estimator_.get_params()["classifier"])

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe

X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = HyperoptEstimator(
    classifier=any_classifier('cla'),
    preprocessing=any_preprocessing('pre'),
    algo=tpe.suggest,
    max_evals=20,
    trial_timeout=30
)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print("Accuracy: %.3f" % acc)
print(model.best_model())