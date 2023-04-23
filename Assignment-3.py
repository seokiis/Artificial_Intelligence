from sklearn import tree
from sklearn import svm
from sklearn import datasets

d = datasets.load_iris()  # iris 데이터셋을 읽고
print(d.DESCR)  # 내용을 출력
for i in range(0, len(d.data)):  # 샘플을 순서대로 출력
    print(i+1, d.data[i], d.target[i])


s = svm.SVC(gamma=0.1, C=10)  # svm 분류 모델 SVC 객체 생성하고
s.fit(d.data, d.target)  # iris 데이터로 학습

res_SVM = s.predict(d.data)
print("SVM: 샘플의 부류는", res_SVM)

correct = [i for i in range(len(res_SVM)) if res_SVM[i] == d.target[i]]
accuracy = len(correct)/len(res_SVM)
print("정확률=", accuracy*100, "%")

################
# 본인 코드 작성
################

clf = tree.DecisionTreeClassifier()
clf = clf.fit(d.data, d.target)

res_tree = clf.predict(d.data)

correct2 = [i for i in range(len(res_tree)) if res_tree[i] == d.target[i]]
accuracy2 = len(correct2)/len(res_tree)
print("정확률2=", accuracy2*100, "%")
