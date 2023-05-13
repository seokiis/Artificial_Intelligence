from sklearn import datasets
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digit.data, digit.target, train_size=0.6)  # 40%를 테스트 집합으로 함

################
# 본인 코드 작성
################

# 1. SVM 모델 생성 및 평가
svm_model = svm.SVC(gamma=0.1, C=10)
svm_scores = cross_val_score(svm_model, x_train, y_train, cv=5)
print("SVM 모델의 5-겹 교차 검증 평균 정확률:", np.mean(svm_scores))

# 2. 결정 트리 모델 생성 및 평가
tree_model = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_model, x_train, y_train, cv=5)
print("결정 트리 모델의 5-겹 교차 검증 평균 정확률:", np.mean(tree_scores))

# 3. 랜덤 포리스트 모델 생성 및 평가
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5)
rf_scores = cross_val_score(rf_model, x_train, y_train, cv=5)
print("랜덤 포리스트 모델의 5-겹 교차 검증 평균 정확률:", np.mean(rf_scores))

# 선택된 모델의 평균 정확률 출력
max_score = max(np.mean(svm_scores), np.mean(tree_scores), np.mean(rf_scores))
if max_score == np.mean(svm_scores):
    selected_model = svm_model
    print("SVM 모델 선택")
elif max_score == np.mean(tree_scores):
    selected_model = tree_model
    print("결정 트리 모델 선택")
else:
    selected_model = rf_model
    print("랜덤 포리스트 모델 선택")

# 선택된 모델의 성능 평가
selected_model.fit(x_train, y_train)
test_score = selected_model.score(x_test, y_test)
print("선택된 모델의 테스트 집합 정확률:", test_score)
