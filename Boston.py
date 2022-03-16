# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 22:46:28 2022

@author: User
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

#Import and Examine
mydata = pd.read_csv('E:/Angeline/Teaching/business analytics/BA-PGDM/2021-2023/Datasets/NN/Prediction/Boston Housing.csv')
print(mydata.head(5))
   
print(mydata.dtypes)



#normalise
min_max_scaler = MinMaxScaler()
mydata[["medv","crim","zn","indus","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]]=min_max_scaler.fit_transform(mydata[["medv","crim","zn","indus","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]])
print(mydata)


#Test Train Split
Y=mydata.medv
X=mydata.drop(columns='medv')
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

#One Hidden layer with 5 nodes
mymodel = MLPRegressor(hidden_layer_sizes=(5),random_state=1, max_iter=1000,solver="lbfgs")
mymodel.fit(X_train, y_train)

print(mymodel)


#Training Error
predictions_train_LR=mymodel.predict(X_train) #predicted values
train_error_LR=mse(y_train,predictions_train_LR) #actual-predicted
print('Training Error:',train_error_LR)

#Test Error
predicted_y = mymodel.predict(X_test)

test_error_LR=mse(y_test,predicted_y)

score_LR=mymodel.score(X_train,y_train) #returns the R^2 score
print('R^2 returned by the  mymodel:',score_LR)
print('Test Error:',test_error_LR)#evaluate the R^2score_