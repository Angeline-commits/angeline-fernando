# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:17:00 2020

@author: Angeline
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
mydata=pd.read_csv('telco.csv')
type(mydata)
print(mydata.shape)
print(mydata.head(5))
print(mydata.dtypes )

df_gender = pd.get_dummies(mydata["Gender"],prefix='Gender',drop_first=True)
mydata['Gender_Coded'] = df_gender
print(mydata.head(5))
Y=mydata.Churn
X=mydata[['Age','Gender_Coded']]
X = sm.add_constant(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
m=sm.Logit(y_train,X_train)
telco_logit=m.fit()
print(telco_logit.summary())
print(telco_logit.summary2())
#Confusion Matrix
print(telco_logit.pred_table())
