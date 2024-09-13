import pandas as pd  
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("diabetes.csv")
dataset.head()
dataset.shape
dataset.describe()
dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()
x=dataset.drop(columns='Outcome' , axis=1)
y = dataset['Outcome']
# print(x)
# print(y)
scaler=StandardScaler()
scaler.fit(x)
standardised_data=scaler.transform(x)
# print(standardised_data)
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
classifyer=svm.SVC(kernel='linear')
classifyer.fit(x_train, y_train)
prediction=classifyer.predict(x_train)
accuracy=accuracy_score(prediction, y_train)
print("Accuracy of trained Dataset --> ", np.multiply(accuracy, 100), "%")
prediction=classifyer.predict(x_test)
accuracy=accuracy_score(prediction, y_test)
print("Accuracy of test Dataset --> ", np.multiply(accuracy, 100), "%")
input=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array=np.asarray(input)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=classifyer.predict(input_data_reshaped)

if (prediction[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')