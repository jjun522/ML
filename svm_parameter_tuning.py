from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import make_classification       # 예제 데이터 생성용

# # --- 2. 예시 데이터 준비 (svm_basic_example.py와 동일한 데이터 사용) ---
# X = np.array([
#     [1, 2], [1.5, 1.8], [2, 1.5], [1.8, 2.2], # 클래스 0 데이터
#     [4, 5], [4.5, 4.8], [5, 4.5], [4.8, 5.2]  # 클래스 1 데이터
# ])
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=0.8, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- RBF 커널 (기본값) ---")
svc_rbf = SVC(kernel='rbf', random_state=42)
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)
print(f"예측:{y_pred_rbf},실제:{y_test},정확도:{accuracy_score(y_test, y_pred_rbf)}")

print("\n--- RBF 커널, 높은 C ---")
svc_rbf_high_c= SVC(kernel='rbf', C= 100 ,random_state=42)
svc_rbf_high_c.fit(X_train, y_train)
y_pred_rbf_high_c = svc_rbf_high_c.predict(X_test)
print(f"예측:{y_pred_rbf_high_c},실제:{y_test},정확도:{accuracy_score(y_test, y_pred_rbf_high_c)}")

print("\n--- RBF 커널, 낮은 C ---")
svc_rbf_low_c = SVC(kernel='rbf', C= 0.01 ,random_state=42)
svc_rbf_low_c.fit(X_train, y_train)
y_pred_rbf_low_c = svc_rbf_low_c.predict(X_test)
print(f"예측:{y_pred_rbf_low_c},실제:{y_test},정확도:{accuracy_score(y_test, y_pred_rbf_low_c)}")

print("\n--- RBF 커널, 낮은 gamma---")
svc_rbf_low_gamma = SVC(kernel='rbf', gamma= 0.01 ,random_state=42)
svc_rbf_low_gamma.fit(X_train, y_train)
y_pred_rbf_low_gamma = svc_rbf_low_gamma.predict(X_test)
print(f"예측:{y_pred_rbf_low_gamma},실제:{y_test},정확도:{accuracy_score(y_test, y_pred_rbf_low_gamma)}")


print("\n--- 다항식 커널 (degree=3) ---")
svc_poly = SVC(kernel='poly', degree= 3, random_state=42)
svc_poly.fit(X_train, y_train)
y_pred_poly = svc_poly.predict(X_test)
print(f"예측:{y_pred_poly},실제:{y_test},정확도:{accuracy_score(y_test, y_pred_poly)}")



