# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:12:07 2020

@author: Angeline
"""
#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

mydata=pd.read_csv('./data/insurance_fraud.csv')
type(mydata)
print(mydata.shape)
print(mydata.head(5))
print(mydata.dtypes)

mydata['fraud'] = mydata['fraud'].astype('category')
mydata['claim_type'] = mydata['claim_type'].astype('category')
mydata['gender'] = mydata['gender'].astype('category')
mydata['edcat'] = mydata['edcat'].astype('category')
mydata['retire'] = mydata['retire'].astype('category')
mydata['marital'] = mydata['marital'].astype('category')
mydata['reside'] = mydata['reside'].astype('category')
mydata['primary_residence'] = mydata['primary_residence'].astype('category')
print(mydata.dtypes)

#Dummy coding
df_new =pd.get_dummies(mydata, columns=['claim_type', 'gender','edcat','retire','marital','reside','primary_residence'],prefix=['claim_type_', 'gender_','edcat_','retire_','marital_','reside_','primary_residence_'],drop_first=True)
print(df_new)
print(df_new.dtypes)

#normalise
min_max_scaler = MinMaxScaler()
df_new[["claim_amount", "coverage","townsize","income"]] = min_max_scaler.fit_transform(df_new[["claim_amount", "coverage","townsize","income"]])
df_new

Y=df_new.fraud
X=df_new[[ 'claim_amount',    'coverage',    'townsize',    'income',    'fraud',    'claim_type__2',    'claim_type__3',    'claim_type__4',    'claim_type__5',    'gender__1',    'edcat__2',    'edcat__3',    'edcat__4',    'edcat__5',    'retire__1',    'marital__1',    'reside__2',    'reside__3',    'reside__4',    'reside__5',    'reside__6',    'reside__7',    'reside__8',    'reside__10',    'primary_residence__1']]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

m = MLPClassifier(solver="adam",alpha=1e-05,hidden_layer_sizes=(1, 15), activation = 'logistic',random_state=1,max_iter=5000)
m.fit(X_train, y_train)
m

#Has it overfitted ?
#Check

#Predicting y for X_val
y_pred = m.predict(X_test)
cm = confusion_matrix(y_pred, y_test)

#Printing the accuracy
print(cm)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))