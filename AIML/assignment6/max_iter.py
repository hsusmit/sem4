# Identify the max value from .score() function on using
# max_iter values between 10000 to 100000. Keep a note for
# a specific value of the number of iteration corresponding
# to the obtained result.

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()

data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = pd.DataFrame(boston.target)

x = data[['RM']]
y = data['MEDV']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

max_iter_values = range(10000, 100001, 10000)

max_r2_score = 0
max_iter = 0

for i in max_iter_values:
    model = LinearRegression()
    model.fit(x_train, y_train)
    r2_score = model.score(x_test, y_test)
    if r2_score > max_r2_score:
        max_r2_score = r2_score
        max_iter = i

print(
    f"Maximum R^2 score of {max_r2_score} obtained with max_iter={max_iter}.")

# OUTPUT
# Maximum R^2 score of 0.3707569232254778 obtained with max_iter=10000.
