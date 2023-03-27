# Identify the max value from .score() function on using
# max_iter values between 10000 to 100000. Keep a note for
# a specific value of the number of iteration corresponding
# to the obtained result.

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

boston = load_boston()

data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = pd.DataFrame(boston.target)

x = data[['RM']]
y = data['MEDV']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

max_iter_values = range(10000, 100001, 10000)

r2_scores = []
for max_iter in max_iter_values:
    model = SGDRegressor(max_iter=max_iter)
    model.fit(x_train, y_train)
    r2_score = model.score(x_test, y_test)
    r2_scores.append(r2_score)

max_r2_score = max(r2_scores)
max_iter_index = r2_scores.index(max_r2_score)
max_iter = max_iter_values[max_iter_index]

print(
    f"Maximum R^2 score of {max_r2_score} obtained with max_iter={max_iter}.")

# OUTPUT
# Maximum R^2 score of 0.30156333023278636 obtained with max_iter=80000.
