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
# max_depth = 1, R^2 score = 0.3602156982888398
# max_depth = 2, R^2 score = 0.6455495710736121
# max_depth = 3, R^2 score = 0.8160292234821611
# max_depth = 4, R^2 score = 0.7306868217984426
# max_depth = 5, R^2 score = 0.885137272531848
# max_depth = 6, R^2 score = 0.8673636472053167
# max_depth = 7, R^2 score = 0.7289399366126473
# max_depth = 8, R^2 score = 0.8860954007733421
# max_depth = 9, R^2 score = 0.8760090718789401
# max_depth = 10, R^2 score = 0.8537065241879563
# Optimum max_depth value = 8

# Random Forest Regressor
# max_depth = 1, R^2 score = 0.4827301126691087
# max_depth = 2, R^2 score = 0.7157243744861153
# max_depth = 3, R^2 score = 0.8328240698982
# max_depth = 4, R^2 score = 0.8429144658172141
# max_depth = 5, R^2 score = 0.8616094916175724
# max_depth = 6, R^2 score = 0.8642816675611302
# max_depth = 7, R^2 score = 0.8775811274928114
# max_depth = 8, R^2 score = 0.8671572670523888
# max_depth = 9, R^2 score = 0.8845548450237113
# max_depth = 10, R^2 score = 0.8618766913306678
# Optimum max_depth value = 9

# Decision Tree Classifier
# max_depth = 1, Accuracy score = 0.8947368421052632
# max_depth = 2, Accuracy score = 0.9298245614035088
# max_depth = 3, Accuracy score = 0.9473684210526315
# max_depth = 4, Accuracy score = 0.9385964912280702
# max_depth = 5, Accuracy score = 0.9473684210526315
# max_depth = 6, Accuracy score = 0.9385964912280702
# max_depth = 7, Accuracy score = 0.9298245614035088
# max_depth = 8, Accuracy score = 0.9385964912280702
# max_depth = 9, Accuracy score = 0.9473684210526315
# max_depth = 10, Accuracy score = 0.9473684210526315
# Optimum max_depth value = 3
