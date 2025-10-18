from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]) # 입력 특성 (10x2 배열)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

x_training_set, x_test_set, y_train_set, y_test_set = train_test_split(
    X,y, test_size = 0.2, random_state = 42
)

model = DecisionTreeClassifier(max_depth=5,random_state=0)

model.fit(x_training_set, y_train_set)

y_pred = model.predict(x_test_set)

print(y_pred)
print(y_test_set)