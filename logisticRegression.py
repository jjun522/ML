from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("minist 데이터셋 로딩중 ..")
mnist = fetch_openml('mnist_784', version=1,parser='auto')
print("끝")

X_data = mnist.data
y_data = mnist.target

test_size = 1/7.0
train_img,test_img,train_label,test_label = train_test_split(X_data,y_data,test_size=test_size,random_state=0)

print(f"학습 이미지 데이터 형태: {train_img.shape}")
print(f"테스트 이미지 데이터 형태: {test_img.shape}")

logistic = LogisticRegression(solver='lbfgs')

logistic.fit(train_img,train_label)

prediction = logistic.predict(test_img)

score = logistic.score(test_img,test_label)
print(score)



