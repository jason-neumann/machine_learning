from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.target)
#print(iris.data)

knn = KNeighborsClassifier(n_neighbors=1)
#print(knn)
knn.fit(iris.data, iris.target)
answer = knn.predict([[4,6,5,3]])
print (iris.target_names[answer])

knn.set_params(n_neighbors = 5)
knn.fit(iris.data, iris.target)
answer = knn.predict([[4,6,5,3]])
print (iris.target_names[answer])

logRegModel = LogisticRegression()
logRegModel.fit(iris.data, iris.target)
answer = logRegModel.predict([[2,4,3,1]])
print (iris.target_names[answer])
answer = logRegModel.predict([[4,6,5,3]])
print (iris.target_names[answer])