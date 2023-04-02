# Using iterations, and the classification/regression models,
# try to identify the optimum max_depth value
# at which the following 3 models give the maximum R2 score.
# 1. Decision Tree Regressor
# 2. Random Forest Regressor
# 3. Decision Tree Classifier

import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load Boston Housing dataset for regression task
boston = load_boston()
data_reg = pd.DataFrame(boston.data, columns=boston.feature_names)
data_reg['target'] = pd.DataFrame(boston.target)

# Load Breast Cancer dataset for classification task
cancer = load_breast_cancer()
data_cls = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_cls['target'] = pd.DataFrame(cancer.target)

# Split the datasets into train and test sets
x_reg = data_reg.drop('target', axis=1)
y_reg = data_reg['target']
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x_reg, y_reg, test_size=0.2, random_state=42)

x_cls = data_cls.drop('target', axis=1)
y_cls = data_cls['target']
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
    x_cls, y_cls, test_size=0.2, random_state=42)

# Decision Tree Regressor
print('Decision Tree Regressor')
r2_scores = []
for depth in range(1, 11):
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(x_train_reg, y_train_reg)
    y_pred_reg = model.predict(x_test_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    r2_scores.append(r2)
    print(f"max_depth = {depth}, R^2 score = {r2}")
print(f"Optimum max_depth value = {r2_scores.index(max(r2_scores))+1}\n")

# Random Forest Regressor
print('Random Forest Regressor')
r2_scores = []
for depth in range(1, 11):
    model = RandomForestRegressor(max_depth=depth)
    model.fit(x_train_reg, y_train_reg)
    y_pred_reg = model.predict(x_test_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    r2_scores.append(r2)
    print(f"max_depth = {depth}, R^2 score = {r2}")
print(f"Optimum max_depth value = {r2_scores.index(max(r2_scores))+1}\n")

# Decision Tree Classifier
print('Decision Tree Classifier')
accuracy_scores = []
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train_cls, y_train_cls)
    y_pred_cls = model.predict(x_test_cls)
    accuracy = model.score(x_test_cls, y_test_cls)
    accuracy_scores.append(accuracy)
    print(f"max_depth = {depth}, Accuracy score = {accuracy}")
print(
    f"Optimum max_depth value = {accuracy_scores.index(max(accuracy_scores))+1}\n")


# OUTPUT
# Decision Tree Regressor
# max_depth = 1, R^2 score = 0.3602156982888397
# max_depth = 2, R^2 score = 0.6455495710736121
# max_depth = 3, R^2 score = 0.8160292234821611
# max_depth = 4, R^2 score = 0.855229172828375
# max_depth = 5, R^2 score = 0.8833565347917995
# max_depth = 6, R^2 score = 0.7176628338469286
# max_depth = 7, R^2 score = 0.8671839668168589
# max_depth = 8, R^2 score = 0.7079869883148955
# max_depth = 9, R^2 score = 0.8806587759779556
# max_depth = 10, R^2 score = 0.6817728268989374
# Optimum max_depth value = 5

# Random Forest Regressor
# max_depth = 1, R^2 score = 0.49460698197557706
# max_depth = 2, R^2 score = 0.7048097827827444
# max_depth = 3, R^2 score = 0.8284738655974604
# max_depth = 4, R^2 score = 0.8453015302998119
# max_depth = 5, R^2 score = 0.8665923018901309
# max_depth = 6, R^2 score = 0.8616453743764854
# max_depth = 7, R^2 score = 0.8887760698005752
# max_depth = 8, R^2 score = 0.879164105646071
# max_depth = 9, R^2 score = 0.8812833122027885
# max_depth = 10, R^2 score = 0.8904026480270838
# Optimum max_depth value = 10

# Decision Tree Classifier
# max_depth = 1, Accuracy score = 0.8947368421052632
# max_depth = 2, Accuracy score = 0.9298245614035088
# max_depth = 3, Accuracy score = 0.9473684210526315
# max_depth = 4, Accuracy score = 0.9298245614035088
# max_depth = 5, Accuracy score = 0.9473684210526315
# max_depth = 6, Accuracy score = 0.9298245614035088
# max_depth = 7, Accuracy score = 0.9473684210526315
# max_depth = 8, Accuracy score = 0.9473684210526315
# max_depth = 9, Accuracy score = 0.9298245614035088
# max_depth = 10, Accuracy score = 0.9385964912280702
# Optimum max_depth value = 3
