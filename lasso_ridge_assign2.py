# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:25:48 2021

@author: asilp
"""

import pandas as pd
import numpy as np
import seaborn as sns
computer_data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign13_lasso\computer_data.csv")
computer_data.columns
computer_data.drop(['Unnamed: 0'],axis=1,inplace=True)
computer_data.describe()
computer_data.dtypes
computer_data=pd.get_dummies(computer_data,prefix=['cd','multi','premium'],columns=['cd','multi','premium'],drop_first=True)
sns.pairplot(computer_data)

x=computer_data.drop(['price'],axis=1)
y=computer_data.iloc[:,[0]]

from sklearn.model_selection import cross_val_score,train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
##linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
mse_linear=cross_val_score(reg,x,y,scoring='neg_mean_squared_error',cv=5)
mean_mse_linear=np.mean(mse_linear)

## ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)
ridge_regressor.best_params_
ridge_regressor.best_score_

##Lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)
lasso_regressor.best_params_
lasso_regressor.best_score_
##model predictions
predict_ridge=ridge_regressor.predict(x_test)
predict_lasso=lasso_regressor.predict(x_test)
predict_lasso=predict_lasso.reshape(1252,1)
##graphical representations
import seaborn as sns
sns.distplot(y_test-predict_ridge)
sns.distplot(y_test-predict_lasso)
