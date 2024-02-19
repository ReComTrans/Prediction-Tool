# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:09:03 2021

@author: krembsler
"""

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from tscv import GapWalkForward
from methods.methods import methodname

def series_to_supervised(data,Treiber, lags,  dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
   # data, Treiber = Exo_trans,Exo_trans.columns.tolist()
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in lags:
		cols.append(df.shift(i))
		names += [(j+'(t-%d)' % ( i)) for j in Treiber]
	# forecast sequence (t, t+1, ... t+n)
	for i in [0]:
		cols.append(df.shift(-i))
		if i == 0:
			names += [(j+'(t)')  for j in Treiber]
		else:
			names += [(j+'(t+%d)' % (i)) for j in Treiber]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def Ridge(Train, ExoTrain, ExoTest, cv=5, window=False, lags = False, scaled_coeff=False):
    
    alphas = 10**np.linspace(10,-2,8000)*0.5
    if lags ==False:
        pass
    else:
        ExoTrain=series_to_supervised(ExoTrain,ExoTrain.columns.tolist(), lags=lags, dropnan=True)
        
    if window ==False:
        cv = cv
    elif window == "rolling" : 
        cv =GapWalkForward(n_splits=cv, gap_size=0, test_size=12)# max_train_size=36 wenn ohne max_train dann nicht block sondern Rolling split
    else:
        print("Window nicht verfügabar. Es wird mit normaler {}-fachen Cross Validation fortgefahren".format(cv))
    #Schleife über Produktgruppen 
    Ind  = ExoTrain.columns.tolist()
    Ind.append("Intercept")
    Ind.append("Regularization Strength Alpha")
    Koeff= pd.DataFrame(index=Ind,columns=[methodname.ridge])
    #Koeff= pd.Series(index=Ind,name="Ridge")
    
    pipe = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('clf', RidgeCV(alphas=alphas,cv=cv,fit_intercept=True))])
    
    #if type(Train) is pd.core.frame.DataFrame and len(Train.columns)==1:
    #    Train = Train[Train.columns[0]]

 
    ridge = pipe.fit(ExoTrain,Train.values.ravel())
    
     
    Fit =  pd.Series(ridge.predict(ExoTrain).ravel(),index=Train.index)
    Pred = pd.Series(ridge.predict(ExoTest).ravel(),index=ExoTest.index)
    
    
    if scaled_coeff==False:
        coeff = ridge.named_steps['clf'].coef_ /ridge.named_steps['scale'].scale_
        intercept =  ridge.named_steps['clf'].intercept_ -   sum(ridge.named_steps['clf'].coef_ * ridge.named_steps['scale'].mean_ / ridge.named_steps['scale'].scale_)
    
        coeff = np.append(coeff,intercept)
        coeff = np.append(coeff,ridge.named_steps['clf'].alpha_)
        Koeff[methodname.ridge]= pd.Series(coeff,index=Ind)

    else:
        coeff = ridge.named_steps['clf'].coef_ 
        intercept =  ridge.named_steps['clf'].intercept_
        
        coeff = np.append(coeff,intercept)
        coeff = np.append(coeff,ridge.named_steps['clf'].alpha_)
        Koeff[methodname.ridge]= pd.Series(coeff,index=Ind)
    
   
    return (Pred,Fit,Koeff)



def Ridge_Szen(Train, ExoTrain, ExoTest, cv=5, window=False, lags = False, scaled_coeff=False):
    
    alphas = 10**np.linspace(10,-2,8000)*0.5
    if lags ==False:
        pass
    else:
        ExoTrain=series_to_supervised(ExoTrain,ExoTrain.columns.tolist(), lags=lags, dropnan=True)
        
    if window ==False:
        cv = cv
    elif window == "rolling" : 
        cv =GapWalkForward(n_splits=cv, gap_size=0, test_size=12)# max_train_size=36 wenn ohne max_train dann nicht block sondern Rolling split
    else:
        print("Window nicht verfügabar. Es wird mit normaler {}-fachen Cross Validation fortgefahren".format(cv))
    #Schleife über Produktgruppen 
    Ind  = ExoTrain.columns.tolist()
    Ind.append("Intercept")
    Koeff= pd.DataFrame(index=Ind,columns=[methodname.ridge])
    #Koeff= pd.Series(index=Ind,name="Ridge")
    
    pipe = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('clf', RidgeCV(alphas=alphas,cv=cv,fit_intercept=True))])
    

 
    ridge = pipe.fit(ExoTrain,Train)
    
     
    Fit =  pd.Series(ridge.predict(ExoTrain).ravel(),index=Train.index)
    #Pred = pd.Series(ridge.predict(ExoTest).ravel(),index=ExoTest.index)
    
    
    if scaled_coeff==False:
        coeff = ridge.named_steps['clf'].coef_ /ridge.named_steps['scale'].scale_
        intercept =  ridge.named_steps['clf'].intercept_ -   sum(ridge.named_steps['clf'].coef_ * ridge.named_steps['scale'].mean_ / ridge.named_steps['scale'].scale_)
    
        Koeff[methodname.ridge]= pd.Series(np.append(coeff,intercept),index=Ind)

    else:
        coeff = ridge.named_steps['clf'].coef_ 
        intercept =  ridge.named_steps['clf'].intercept_
        
        Koeff[methodname.ridge]= pd.Series(np.append(coeff,intercept),index=Ind)
    
   
    return (ridge,Fit,Koeff)