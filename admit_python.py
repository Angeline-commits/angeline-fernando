# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:17:00 2020
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
@author: Angeline
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

mydata=pd.read_csv('./data/admit.csv')
type(mydata)
print(mydata.shape)
print(mydata.head(5))
print(mydata.dtypes)

mydata['admit'] = mydata['admit'].astype('category')
mydata['gender'] = mydata['gender'].astype('category')
mydata['rank'] = mydata['rank'].astype('category')

#Dummy coding
df_new =pd.get_dummies(mydata, columns=['gender', 'rank'],prefix=['gender_', 'rank_'],drop_first=True)

print(df_new)
Y=df_new.admit
X=df_new[['gre','gpa','gender__Male','rank__2','rank__3','rank__4']]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

log_model = LogisticRegression(C=1e6,solver='newton-cg')
m = log_model.fit(X_train,y_train)
y_pred=m.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Change cut-offs

#Predict the class probability
y_pred=m.predict_proba(X_test)
y_pred
#Second index checks the probaility of class 1
y_prob_class1= y_pred[:,1]
y_prob_class1
y_test

#For a cut-off of 0.5
y_pred=[1 if i > 0.5 else 0 for i in y_prob_class1]
y_pred
#convert to categorical -to compare with original test data
type(y_pred)
y_pred = pd.Series(y_pred)
#y_pred= pd.Categorical(y_pred)
type(y_pred)
type(y_test)
print(confusion_matrix(y_test, y_pred))

#confusion_matrix(y_test y_pred)
#Varying threshold/cut-offs
cutoffs = [0.21,0.1,0.5,0.6,0.9]
for j in cutoffs:
   y_pred=[1 if i > j else 0 for i in y_prob_class1]
   print(confusion_matrix(y_test, y_pred))
   print(accuracy_score(y_test, y_pred))
  
