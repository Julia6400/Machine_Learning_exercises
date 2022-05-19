from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

wine = load_wine()
features, target = wine.data, wine.target
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = DummyClassifier()
model.fit(X_train, y_train)
preds= model.predict(X_test)

print(model.score(preds, y_test))

filename = 'dummy_model.sav'
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)

scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

n_neighbors = list(range(1, 11))
test_score = []
train_score = []

for i in n_neighbors:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_std, y_train)
    test_score.append(round(model.score(X=X_train_std, y=y_train), 3))
    train_score.append(round(model.score(X=X_test_std, y=y_test), 3))

plt.figure(figsize=(10,5))

sns.lineplot(x=n_neighbors, y=test_score, label="Test data")
sns.lineplot(x=n_neighbors, y=train_score, label="Train data")

plt.title("Number of neighbors and score")
plt.legend()
plt.grid()
plt.show()