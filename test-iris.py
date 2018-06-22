
# import load_iris function from datasets module# impor
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X, y)
print(knn.predict([[3, 5, 4, 2]]))
