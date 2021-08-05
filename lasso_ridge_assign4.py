# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:25:48 2021

@author: asilp
"""

import pandas as pd
import numpy as np
import seaborn as sns
life_data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign13_lasso\Life_expectencey_LR.csv")
life_data.columns

life_data.describe()
life_data.dtypes
life_data.isnull().sum()
life_data.drop(['Country'],axis=1,inplace=True)
###label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
life_data['Status']=le.fit_transform(life_data['Status'])
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
life_data['Life_expectancy']=mean_imputer.fit_transform(life_data[['Life_expectancy']])
life_data['Adult_Mortality']=mean_imputer.fit_transform(life_data[['Adult_Mortality']])
life_data['Alcohol']=mean_imputer.fit_transform(life_data[['Alcohol']])
life_data['Hepatitis_B']=mean_imputer.fit_transform(life_data[['Hepatitis_B']])
life_data['BMI']=mean_imputer.fit_transform(life_data[['BMI']])
life_data['Polio']=mean_imputer.fit_transform(life_data[['Polio']])
life_data['Total_expenditure']=mean_imputer.fit_transform(life_data[['Total_expenditure']])
life_data['Diphtheria']=mean_imputer.fit_transform(life_data[['Diphtheria']])
life_data['GDP']=mean_imputer.fit_transform(life_data[['GDP']])
life_data['Population']=mean_imputer.fit_transform(life_data[['Population']])
life_data['thinness']=mean_imputer.fit_transform(life_data[['thinness']])
life_data['thinness_yr']=mean_imputer.fit_transform(life_data[['thinness_yr']])
life_data['Income_composition']=mean_imputer.fit_transform(life_data[['Income_composition']])
life_data['Schooling']=mean_imputer.fit_transform(life_data[['Schooling']])
life_data.isnull().sum()


x=life_data.drop(['Life_expectancy'],axis=1)
y=life_data.iloc[:,[2]]

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
predict_lasso=predict_lasso.reshape(588,1)
##graphical representations
import seaborn as sns
sns.distplot(y_test-predict_ridge)
sns.distplot(y_test-predict_lasso)
