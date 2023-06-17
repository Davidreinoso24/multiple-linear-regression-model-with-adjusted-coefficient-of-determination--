# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:00:22 2023

@author: herna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cargar el dataset
dataset=pd.read_csv("Multiple Linear Regression_50_Startups.csv")
X=dataset.iloc[:,4].values
Y= dataset.iloc[:,:-1].values

# codificacion de datos 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
dum=ColumnTransformer( [("one_hot_encoder",OneHotEncoder(categories="auto"),[3])], remainder="passthrough")
Y=np.array(dum.fit_transform(Y),dtype=np.float)
Y=Y[:,1:]

#divicion del dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y , test_size=0.2, random_state=0)

#Modelo de regresion lineal multiple con el conjunto train 
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(Y_train,X_train)

#prediccion
X_predic=regression.predict(Y_test)

#validacion automatica del modelo eliminacion hacia atras 
import statsmodels.api as sm 
Y=np.append(arr=np.ones((50,1)).astype(int),values=Y, axis=1)
def eliminacionhaciaatras(Y,SL):
    numVar= len(Y[0])
    temp= np.zeros((50,6)).astype(int)
    for i in range(0,numVar):
        regression_ols=sm.OLS(X,Y.tolist()).fit()
        maxvar=max(regression_ols.pvalues).astype(float)
        adjR_before=regression_ols.rsquared_adj.astype(float )
        if maxvar>SL:
            for j in range(0,numVar-i):
                if(regression_ols.pvalues[j].astype(float)==maxvar):
                    temp[:,j]=Y[:,j]
                    Y= np.delete(Y,j,1)
                    temp_regression=sm.OLS(X,Y.tolist()).fit()
                    adjR_after= temp_regression.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        Y_back=np.hstack((Y, temp[:,[0,j]]))
                        Y_back=np.delete(Y_back,j,1)
                        regression_ols.summary()
                        return Y_back
                    else:
                        continue
                
    regression_ols.summary()
    return Y
                
SL=0.05
Y_opt=Y[:,[0,1,2,3,4,5]]
Y_model=eliminacionhaciaatras(Y_opt, SL)







    
    










 
