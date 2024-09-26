import numpy as np
import pandas as pd

dataset = pd.read_csv("iris.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)

new_test_point = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = classifier.predict(new_test_point)
print(f"\n Predicted class: {prediction[0]}")

'''
OUTPUT

mlm@mlm-ThinkCentre-E73:~/Rojin_Roy/RO_DATA_SCIENCE$ python3 knn.py
              precision    recall  f1-score   support

      Setosa       0.75      1.00      0.86         6
  Versicolor       0.92      0.79      0.85        14
   Virginica       0.90      0.90      0.90        10

    accuracy                           0.87        30
   macro avg       0.86      0.90      0.87        30
weighted avg       0.88      0.87      0.87        30

Accuracy :  0.8666666666666667
   Real Values Predicted Values
0   Versicolor       Versicolor
1   Versicolor       Versicolor
2   Versicolor       Versicolor
3       Setosa           Setosa
4    Virginica        Virginica
5    Virginica        Virginica
6    Virginica        Virginica
7    Virginica        Virginica
8   Versicolor           Setosa
9       Setosa           Setosa
10  Versicolor       Versicolor
11   Virginica        Virginica
12   Virginica        Virginica
13      Setosa           Setosa
14   Virginica        Virginica
15      Setosa           Setosa
16  Versicolor       Versicolor
17  Versicolor           Setosa
18      Setosa           Setosa
19  Versicolor       Versicolor
20  Versicolor        Virginica
21  Versicolor       Versicolor
22   Virginica       Versicolor
23  Versicolor       Versicolor
24   Virginica        Virginica
25  Versicolor       Versicolor
26      Setosa           Setosa
27  Versicolor       Versicolor
28   Virginica        Virginica
29  Versicolor       Versicolor

 Predicted class: Setosa
'''
