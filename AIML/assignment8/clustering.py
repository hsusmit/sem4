# On the iris dataset, performs the kmeans clustering technique.
# Use any 2 features to perform the clustering.
# Plot the clusters and cluster centers.

# Next, perform the Decision Tree clssification on the iris dataset.
# Determine the r2 score and mean squared error for the classifier.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

iris = load_iris()

# Use the sepal length and width features
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], marker='*', s=300, color='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means clustering on the Iris dataset')
plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R2 score:', r2)
print('Mean squared error:', mse)
