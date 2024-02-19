# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:25:12 2021

@author: Spiegel
"""
import numpy as np
import pandas as pd
import os
from scipy.linalg import toeplitz
from scipy import stats
import math
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, linear_model 
import statsmodels.api as sm
from methods.methods import methodname

def series_to_supervised(data, Treiber, lags,  dropnan=True):
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



def MLR(Train, ExoTrain, ExoTest, lags = False, scaled_coeff=False):
    
    
    if lags ==False:
        pass
    else:
        ExoTrain=series_to_supervised(ExoTrain,ExoTrain.columns.tolist(), lags=lags, dropnan=True)
    #Schleife über Produktgruppen 
    Ind  = ExoTrain.columns.tolist()
    Ind.append("Intercept")
    Koeff= pd.DataFrame(index=Ind,columns=[methodname.mlr])
    
    pipe = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('clf',   LinearRegression(fit_intercept=True))])
    
    #if type(Train) is pd.core.frame.DataFrame and len(Train.columns)==1:
    #    Train = Train[Train.columns[0]]
 
    ols = pipe.fit(ExoTrain,Train.values.ravel())
    
   
    
    Fit =  pd.Series(ols.predict(ExoTrain).ravel(),index=Train.index)
    Pred = pd.Series(ols.predict(ExoTest).ravel(),index=ExoTest.index)
    
    
    if scaled_coeff==False:
        coeff = ols.named_steps['clf'].coef_ /ols.named_steps['scale'].scale_
        intercept =  ols.named_steps['clf'].intercept_ -  sum(ols.named_steps['clf'].coef_ * ols.named_steps['scale'].mean_ / ols.named_steps['scale'].scale_)
    
        Koeff[methodname.mlr]= pd.Series(np.append(coeff,intercept),index=Ind)

    else:
        coeff = ols.named_steps['clf'].coef_ 
        intercept =  ols.named_steps['clf'].intercept_
        
        Koeff[methodname.mlr]= pd.Series(np.append(coeff,intercept),index=Ind)
    
   
    return (Pred,Fit,Koeff)
    
    
def MLR_stats(Train, ExoTrain, ExoTest, lags = False, scaled_coeff=False):
    
   
  
    if lags ==False:
        pass
    else:
        ExoTrain=series_to_supervised(ExoTrain,ExoTrain.columns.tolist(), lags=lags, dropnan=True)
    #Schleife über Produktgruppen 
    Ind  = ExoTrain.columns.tolist()
    Ind.append("Intercept")
    Koeff= pd.DataFrame(index=Ind,columns=[methodname.mlr])
    
    ExoTrain_non = sm.tools.tools.add_constant(ExoTrain,has_constant='add')
    ExoTest = sm.tools.tools.add_constant(ExoTest,has_constant='add')
    model = sm.OLS(Train,ExoTrain_non,hasconst=False)

    ols =model.fit()
   
    
    Fit =  pd.Series(ols.predict(ExoTrain_non).ravel(),index=Train.index)
    Pred = pd.Series(ols.predict(ExoTest).ravel(),index=ExoTest.index)
    
   
    
    if scaled_coeff==False:
         Koeff[methodname.mlr]= pd.Series(np.append(ols.params[1:],ols.params[0]),index=Ind)

    elif scaled_coeff==True:
        
         ExoTrain_scaled =  ExoTrain.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
         ExoTrain_scaled = sm.tools.tools.add_constant(ExoTrain_scaled)


         model = sm.OLS(Train,ExoTrain_scaled ,hasconst=False)
    
         ols =model.fit()
        
         Koeff[methodname.mlr]= pd.Series(np.append(ols.params[1:],ols.params[0]),index=Ind)

    else:
        print("scaled_coeff muss True oder False sein")
    
   
    return (Pred,Fit,Koeff,ols)
 










 
# def Regressionen(dfAlle, Exo, U, PNum, timeCut, tUnderP): 
#     #Aufteilen in Test und Trainingsdaten
#     #bei Treibern Bevölkerung und Pendlern
#     #sRtrain = dfAlle[ (dfAlle.index <= timeCut) & (dfAlle.index >= tUnderP)]#tUnderP für Pendler
#     #sonst diese Aufteilung
#     sRtrain = dfAlle[dfAlle.index <= timeCut]
 
#     sRtest  = dfAlle[ dfAlle.index >  timeCut ]
 
#     sRpredOls = pd.DataFrame(index=pd.DatetimeIndex(sRtest.index.values,freq=sRtrain.index.inferred_freq))
#     sRpredRlm = pd.DataFrame(index=pd.DatetimeIndex(sRtest.index.values,freq=sRtrain.index.inferred_freq))
#     sRpredGls = pd.DataFrame(index=pd.DatetimeIndex(sRtest.index.values,freq=sRtrain.index.inferred_freq))
 
#     #ändert nichts an der Darstellung und des Outputs, gab aber eine Warnung, dass Frequenz nicht bereitgestellt wurde. 
#     sRtrain.index = pd.DatetimeIndex(sRtrain.index.values,freq=sRtrain.index.inferred_freq)

#     Exo= Exo.astype(float)
#     #bei Treibern Bevölkerung und Pendler
#     #ExoTrain = Exo[ (Exo.index <= timeCut) & (Exo.index >= tUnderP)] #tUnderP für Pendler
#     #sonst diese Aufteilung
#     ExoTrain = Exo[Exo.index <= timeCut]
     
#     ExoTest  = Exo[Exo.index >  timeCut]
     
#     ## Standardisierung der Treiber
#     std_scale = preprocessing.StandardScaler().fit(ExoTrain)
#     ExoTrain = pd.DataFrame(data=std_scale.transform(ExoTrain),columns=ExoTrain.columns,index=ExoTrain.index)
#     ExoTest = pd.DataFrame(data=std_scale.transform(ExoTest),columns=ExoTest.columns,index=ExoTest.index)

#     #Standardisierrung vom Umsatz (reshape muss gemacht werden da sonst nicht mit OLS Befehl konform)
#     std_scale_y = preprocessing.StandardScaler().fit(sRtrain[PNum].values.reshape(-1,1))
#     sRtrainScaled = pd.DataFrame(data=std_scale_y.transform(sRtrain[PNum].values.reshape(-1,1)),index=sRtrain.index)
     
#     ols= OLS(sRtrainScaled,ExoTrain,hasconst=True,missing='drop').fit()
#     model = pd.Series(std_scale_y.inverse_transform(ols.predict(ExoTrain)),index=sRtrain.index)
#     sRpredOls[PNum] = pd.Series(std_scale_y.inverse_transform(ols.predict(ExoTest)),index=sRtest.index)
#     rlm= RLM(sRtrainScaled,ExoTrain,hasconst=True,missing='drop').fit()
#     model2 = pd.Series(std_scale_y.inverse_transform(rlm.predict(ExoTrain)),index=sRtrain.index)
#     sRpredRlm[PNum] = pd.Series(std_scale_y.inverse_transform(rlm.predict(ExoTest)),index=sRtest.index)
         
#     ols_resid = ols.resid
#     res_fit = OLS(list(ols_resid[1:]), list(ols_resid[:-1])).fit()
#     rho = res_fit.params
     
#     order = toeplitz(np.arange(len(sRtrain)))
#     sigma = rho**order
     
#     gls= GLS(sRtrainScaled,ExoTrain,sigma=sigma,hasconst=True,missing='drop').fit()
#     model3 = pd.Series(std_scale_y.inverse_transform(gls.predict(ExoTrain)),index=sRtrain.index)
#     sRpredGls[PNum] = pd.Series(std_scale_y.inverse_transform(gls.predict(ExoTest)),index=sRtest.index)
         
#     return 
 