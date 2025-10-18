# --- 목적 및 활용 ---
# 목적: 손글씨 숫자 이미지(MNIST 데이터셋)를 보고 0부터 9까지 어떤 숫자인지 분류하는 로지스틱 회귀 모델을 학습시킵니다.
# 활용: 이미지 분류(숫자, 간단한 객체), 텍스트 카테고리 분류 등 다중 클래스 분류 문제에 적용될 수 있습니다.

# --- 1. 필요한 라이브러리 가져오기 ---
from sklearn.datasets import fetch_openml             # 공개 데이터셋 로더
from sklearn.model_selection import train_test_split  # 데이터 분할 함수
from sklearn.linear_model import LogisticRegression     # 로지스틱 회귀 모델

# --- 2. MNIST 데이터셋 불러오기 ---
# 인터넷에서 MNIST 손글씨 숫자 데이터셋(이미지 픽셀값 + 정답 숫자)을 다운로드합니다.
print("MNIST 데이터셋 로딩 중...")
mnist = fetch_openml('mnist_784', version=1, parser='auto') # parser='auto'는 데이터 형식 자동 인식
print("MNIST 데이터셋 로딩 완료.")

# 이미지 데이터(X)와 레이블(y) 분리
X_data = mnist.data
y_data = mnist.target

# --- 3. 데이터를 학습용과 테스트용으로 나누기 ---
# [cite_start]전체 데이터를 약 6/7은 학습용, 1/7은 테스트용으로 분할합니다 [cite: 5935-5936].
# random_state=0 : 재현성을 위해 데이터 섞는 방식을 고정합니다.
test_size = 1/7.0
train_img, test_img, train_label, test_label = train_test_split(
    X_data, y_data, test_size=test_size, random_state=0
)

# 분할된 데이터의 크기(형태) 확인
print(f"학습 이미지 데이터 형태: {train_img.shape}") # (샘플 수, 픽셀 수)
print(f"테스트 이미지 데이터 형태: {test_img.shape}") # (샘플 수, 픽셀 수)

# --- 4. 로지스틱 회귀 모델 생성 및 학습 ---
# [cite_start]모델 객체 생성. solver='lbfgs'는 최적화 알고리즘 지정 [cite: 5945-5946].
# 다중 클래스 분류는 내부적으로 One-vs-Rest 방식으로 처리됩니다.
logistic = LogisticRegression(solver='lbfgs', max_iter=1000) # max_iter 늘려서 수렴 보장

# 학습 데이터로 모델 학습
print("모델 학습 시작...")
logistic.fit(train_img, train_label)
print("모델 학습 완료.")

# --- 5. 학습된 모델로 예측하기 ---
# [cite_start]테스트 데이터 일부(처음 10개) 예측 [cite: 5953-5954]
prediction_10 = logistic.predict(test_img[0:10])
print("처음 10개 테스트 이미지 예측 결과:", prediction_10)

# [cite_start]전체 테스트 데이터 예측 [cite: 5955]
print("전체 테스트 데이터 예측 시작...")
predictions = logistic.predict(test_img)
print("전체 테스트 데이터 예측 완료.")

# --- 6. 모델 성능 평가 ---
# [cite_start]테스트 데이터와 테스트 레이블을 사용하여 모델의 정확도(accuracy)를 계산합니다 [cite: 5959-5960].
score = logistic.score(test_img, test_label)
print(f"모델 정확도: {score:.4f}") # 정확도를 소수점 4자리까지 출력