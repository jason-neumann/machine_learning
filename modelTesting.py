from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
answers = knn.predict(X_test)
print(knn.get_params())
print(accuracy_score(y_test, answers))