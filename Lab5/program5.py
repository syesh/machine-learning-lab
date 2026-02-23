import numpy as np
from sklearn.neighbors import KNeighborsClassifier
data = np.random.rand(100).reshape(-1,1)
labels=np.array(["Class1" if x<= 0.5 else "Class2" for x in data [:50]])
train_data,train_labels=data[:50],labels
test_data=data[50:]
k_values = [1,2,3,4,5,20,30]
for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(train_data,train_labels)
  predictions = knn.predict(test_data)
  print(f"Result for k = {k}:")
  print(predictions,"\n")
