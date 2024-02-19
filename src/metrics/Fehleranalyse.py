# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:08:36 2020

@author: Spiegel
"""

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from math import sqrt

#class Fehleranalyse:
    #y_true=y_true.iloc[:,0]
    #y_pred=y_pred.iloc[:,0]
    
def metric_MAE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #mittlere absolute Fehler            
        MAE = mean_absolute_error(y_true,y_pred)
        MAE = round(MAE, 4)
    except:
        MAE = 'MAE konnte nicht berechnet werden'
    return MAE

def metric_MAE_pj(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        MAE = np.abs(np.sum(y_true) - np.sum(y_pred))
        MAE = round(MAE, 4)
        if np.isnan(MAE):
            raise
    except:
        MAE = 'MAE konnte nicht berechnet werden'
    return MAE

def metric_MAE_separate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = np.array([])
    for el_tr, el_pr in zip(y_true, y_pred):
        try:
            mape = np.abs(el_tr - el_pr)
            mape = round(mape, 4)
            if np.isnan(mape):
                raise
            out = np.append(out, mape)
        except: 
            out = np.append(out, 'MAE konnte nicht berechnet werden')
    try: 
        min = np.min(out)
        max = np.max(out)
    except: 
        min = 'MAE konnte nicht berechnet werden'
        max = 'MAE konnte nicht berechnet werden'
    return out, min, max

def metric_MSE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #mittlere quadratische Fehler
        MSE = mean_squared_error(y_true,y_pred)
        MSE = round(MSE, 4)
    except:
        MSE = 'MSE konnte nicht berechnet werden'
    return MSE

def metric_SMAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #SMAPE als alternative zum MAPE (wenn 0en in den Testdaten drin sind)
        #es skaliert den absoluten prozentualen über die Summe von Prognose und beobachteten Wert
        #symmetrischen mittleren absoluten prozentualen Fehlers
        SMAPE = np.mean(np.abs(y_true - y_pred)*((np.abs(y_true) + np.abs(y_pred))*0.5)**(-1))*100
        SMAPE = round(SMAPE, 4)
        if np.isnan(SMAPE):
            raise
    except:
        SMAPE = 'SMAPE konnte nicht berechnet werden'
    return SMAPE

def metric_SMAPE_pj(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        SMAPE = 100*np.abs(np.sum(y_true) - np.sum(y_pred))*((np.abs(np.sum(y_true)) + np.abs(np.sum(y_pred)))*0.5)**(-1)
        SMAPE = round(SMAPE, 4)
        if np.isnan(SMAPE):
            raise
    except:
        SMAPE = 'SMAPE konnte nicht berechnet werden'
    return SMAPE

def metric_SMAPE_separate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = np.array([])
    for el_tr, el_pr in zip(y_true, y_pred):
        try:
            smape = np.abs(el_tr - el_pr)*(np.abs(el_tr) + np.abs(el_pr))**(-1)*200
            smape = round(smape, 4)
            if np.isnan(smape):
                raise
            out = np.append(out, smape)
        except: 
            out = np.append(out, 'SMAPE konnte nicht berechnet werden')
    try: 
        min = np.min(out)
        max = np.max(out)
    except: 
        min = 'SMAPE konnte nicht berechnet werden'
        max = 'SMAPE konnte nicht berechnet werden'
    return out, min, max

def metric_RMSE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #Wurzel des mittlere quadratischen Fehlers
        RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
        RMSE = round(RMSE, 4)
    except:
        RMSE = 'RMSE konnte nicht berechnet werden'
    return RMSE

def metric_MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #monatliche mittleren absolut prozentualen Fehler
        MAPE = np.mean(np.abs((y_true - y_pred)*(y_true)**(-1)))*100
        #MAPE = np.mean(np.abs((y_true - y_pred)/(y_true)))*100
        MAPE = round(MAPE, 4)
        if np.isnan(MAPE):
            raise
    except:
        MAPE = 'monatliche MAPE konnte nicht berechnet werden'
    return MAPE

def metric_MAPE_pj(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #jährliche mittleren absolut prozentualen Fehler
        #MAPE = np.mean(np.abs((y_true - y_pred)*(y_true)**(-1)))*100
        MAPE_pj = np.mean(np.abs((sum(y_true) - sum(y_pred))*(sum(y_true))**(-1)))*100
        MAPE_pj = round(MAPE_pj, 4)
        if np.isnan(MAPE_pj):
            raise
    except:
        MAPE_pj = 'jährliche MAPE konnte nicht berechnet werden'
    return MAPE_pj

def metric_MAPE_separate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = np.array([])
    for el_tr, el_pr in zip(y_true, y_pred):
        try:
            mape = np.abs((el_tr - el_pr)*(el_tr)**(-1))*100
            mape = round(mape, 4)
            if np.isnan(mape):
                raise
            out = np.append(out, mape)
        except: 
            out = np.append(out, 'MAPE konnte nicht berechnet werden')
    try: 
        min = np.min(out)
        max = np.max(out)
    except:
        min = 'MAPE konnte nicht berechnet werden'
        max = 'MAPE konnte nicht berechnet werden'
    return out, min, max

def metric_MFE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #mittlere Prognose Fehler
        MFE = np.mean(y_pred - y_true)
        MFE = round(MFE, 4)
        if np.isnan(MFE):
            raise
    except:
        MFE = 'MFE konnte nicht berechnet werden'
    return MFE

def metric_rel_MFE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        #mittlere Prognose Fehler
        MFE = np.mean((y_pred - y_true) / y_true)*100
        MFE = round(MFE, 4)
        if np.isnan(MFE):
            raise
    except:
        MFE = 'MFE konnte nicht berechnet werden'
    return MFE

def metric_total_bias(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        bias = np.sum(y_pred - y_true)
        bias = round(bias, 4)
        if np.isnan(bias):
            raise
    except:
        bias = 'MFE konnte nicht berechnet werden'
    return bias

def metric_rel_total_bias(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        bias = np.sum(y_pred - y_true)/np.sum(y_true)*100
        bias = round(bias, 4)
        if np.isnan(bias):
            raise
    except:
        bias = 'MFE konnte nicht berechnet werden'
    return bias

def metric_bias_separate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = np.array([])
    for el_tr, el_pr in zip(y_true, y_pred):
        try:
            bias = el_pr - el_tr
            bias = round(bias, 4)
            if np.isnan(bias):
                raise
            out = np.append(out, bias)
        except: 
            out = np.append(out, 'MFE konnte nicht berechnet werden')
    return out

def metric_rel_bias_separate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = np.array([])
    for el_tr, el_pr in zip(y_true, y_pred):
        try:
            bias = ((el_pr - el_tr)*(el_tr)**(-1))*100
            bias = round(bias, 4)
            if np.isnan(bias):
                raise
            out = np.append(out, bias)
        except: 
            out = np.append(out, 'MFE konnte nicht berechnet werden')
    return out



def metric_NMSE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        MSE = mean_squared_error(y_true,y_pred)
        #NMSE normalisiert den erhaltenen MSE nach dividieren durch der Test Varianz.
        #ausgewogenes Fehlermaß und sehr effektiv bei Beurteilung der Prognosegenauigkeit eines Modells
        #normalisierter mittlere quadratischer Fehler
        NMSE = MSE*(np.sum((y_true - np.mean(y_true))**2)*(len(y_true)-1)**(-1))**(-1)
        NMSE = round(NMSE, 4)
    except:
        NMSE = 'NMSE konnte nicht berechnet werden'
    return NMSE

def metric_TUK(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
        #normalisiertes Maß vom totalen Prognosefehler
        #TUK Theils Ungleichheitskoeffizient (Theil_u_statistic)
    Fehler = y_true-y_pred
    mfe = np.sqrt(np.mean(y_pred**2))
    mse = np.sqrt(np.mean(y_true**2))
    rmse = np.sqrt(np.mean(Fehler**2))
    TUK = rmse*(mfe+mse)**(-1)
    return TUK
