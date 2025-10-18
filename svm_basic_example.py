from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np


X = np.array([
    [1, 2], [1.5, 1.8], [2, 1.5], [1.8, 2.2], # 클래스 0 데이터
    [4, 5], [4.5, 4.8], [5, 4.5], [4.8, 5.2]  # 클래스 1 데이터
])
# y: 각 데이터 샘플의 정답 클래스(레이블). (0 또는 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("선형 커널 svm 결과 ")
print("테스트 데이터 예측값",y_pred)
print("테스트 데이터 실제 정답",y_test)

print("\n혼동행렬")
print(confusion_matrix(y_test, y_pred))
print("\n분류 리포트 ")
print(classification_report(y_test, y_pred))