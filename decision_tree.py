# --- 라이브러리 가져오기 ---
from sklearn.model_selection import train_test_split # 데이터 분할용
from sklearn.tree import DecisionTreeClassifier      # 결정 트리 분류 모델
import numpy as np                                   # 배열 처리용

# --- 예시 데이터 준비 (X: 특성, y: 클래스 레이블) ---
# 실제로는 데이터를 파일 등에서 불러옵니다.
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]) # 10개 샘플, 2개 특성
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])             # 각 샘플의 클래스 (0 또는 1)

# --- 데이터를 학습용(80%)과 테스트용(20%)으로 분할 ---
# random_state=42 : 코드를 다시 실행해도 항상 동일하게 데이터를 나누기 위함 (결과 재현성)
x_training_set, x_test_set, y_train_set, y_test_set = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 결정 트리 분류 모델 생성 ---
# max_depth=5 : 트리의 최대 깊이를 5로 제한 (과적합 방지)
# random_state=0 : 모델 생성 시 무작위성을 제어하여 결과 재현성 확보
model = DecisionTreeClassifier(max_depth=5, random_state=0)

# --- 모델 학습 ---
# 학습 데이터를 사용하여 분류 규칙(트리 구조)을 학습
model.fit(x_training_set, y_train_set)

# --- 테스트 데이터로 예측 수행 ---
# 학습된 모델을 사용하여 테스트 데이터의 클래스를 예측
y_pred = model.predict(x_test_set)

# --- 결과 출력 ---
# 모델의 예측 결과와 실제 정답을 비교하여 성능 확인
print("테스트 데이터 예측 결과:", y_pred)
print("테스트 데이터 실제 정답:", y_test_set)