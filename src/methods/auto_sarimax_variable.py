# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:10:33 2021
@author: krembsler
"""


import pandas as pd
from pmdarima.arima import auto_arima, ndiffs
import numpy as np
import pmdarima

from dateutil.relativedelta import relativedelta   

from joblib import Parallel, delayed

def sub_lists (l): 
    base = []   
    lists = [base] 
    for i in range(len(l)): 
        orig = lists[:] 
        new = l[i] 
        for j in range(len(lists)): 
            lists[j] = lists[j] + [new] 
        lists = orig + lists 
    return lists 



    


def AutoSARIMAX(Train, ExoTrain=None, ExoTest=None, Freq="MS", seasonal = True, Season = 12, 
                Steps=12, Kriterium="aic", Trend="Alle", Test="Alle", Methode="powell", 
                Optimierung=True, Parallelisierung=True, max_order=5):
    #Definieren der Listen    
      
    #Datum für Prediction Festlegen
    if Freq=="MS":
        Erster = Train.index[-1]+relativedelta(months=1)
        Letzter =  Train.index[-1]+relativedelta(months=Steps)
    elif Freq=="D":
        Erster = Train.index[-1]+relativedelta(days=1)
        Letzter =  Train.index[-1]+relativedelta(days=Steps)
    elif Freq=="W":
        Erster = Train.index[-1]+relativedelta(weeks=1)
        Letzter =  Train.index[-1]+relativedelta(weeks=Steps)
    else:
        print("Freq muss MS, D oder W sein")                                             
             
              
    if Trend == "Alle":
        Trend = [None,"c","t","ct"]
    else:
        Trend = [Trend]
    if Test == "Alle":
        if ndiffs(Train, test="adf") == ndiffs(Train, test="pp"):
           Test=["adf"] 
        else:
           Test = ["adf", "pp"] 
    else:
         Test = [Test] 
    #Paralellisierter Prozess für jedes Subset von Treibern -> beste p,d,q u. P,D,Q + besten Trend finden 
    def Process(ExoNames):
        #Leere Listen erstellen um Ergebnisse für alle Trends u. Tests abzuspeichern
        results=[]
        Krit=[]
        Trend_list=[]
        Pred=[]
        Fit=[]
        #Schleife für AIC, AICC und BIC
        if Kriterium != "oob":
            #Falls Alle werden zudem alle Trends probiert    
              for test in Test:                    
                    for T in Trend:
                        #Spezialfall keine Exogenen Variablen
                        if ExoNames == []:
                            model = auto_arima(Train, start_p=2, start_q=2, start_d=0,
                                  test=test,       # use adftest to find optimal 'd'
                                  max_p=3, max_q=3,max_d=3, # maximum p and q
                                  m=Season,         # frequency of series
                                  # let model determine 'd'
                                  trend=T,
                                  information_criterion=Kriterium,
                                  seasonal= seasonal,   # No Seasonality
                                  seasonal_test = "ch",
                                  start_P=1, start_Q=1, start_D=0, method=Methode,
                                  max_P=3, max_Q=3,max_D=3,
                                  trace=True,with_intercept='auto',
                                  error_action='trace',  max_order=max_order,
                                  suppress_warnings=True, return_valid_fits =False,
                                  stepwise=True, approximation=False,n_jobs=1)
                            #Speichern des jeweiligen Kriteriums in der Liste
                            result=model.fit(Train)
                            if Kriterium=="aic":
                                Krit.append(result.aic())
                            elif Kriterium=="aicc":
                                Krit.append(result.aicc())
                            elif Kriterium=="bic":
                                Krit.append(result.bic())
                            else:
                                print("Kriterium nicht verfügbar")
                            #Speichern der Ergebnisse in den Listen    
                            Trend_list.append(str(T))
                            # Treiber.append(str(ExoNames))
                            Pred.append(pd.DataFrame(result.predict(Steps),index=pd.date_range(start=Erster,end=Letzter,freq=Freq)))
                            Fit.append(pd.DataFrame(result.predict_in_sample(),index=Train.index) )
                            
                        #Wenn Exo nicht leer -> Konkruent zu oben
                        else:
                            model= auto_arima(Train,ExoTrain[ExoNames], start_p=2, start_q=2, start_d=0,
                                  test=test,       # use adftest to find optimal 'd'
                                  max_p=3, max_q=3,max_d=2, # maximum p and q
                                  m=Season,         # frequency of series
                                  # let model determine 'd'
                                  trend=T,
                                  information_criterion=Kriterium,
                                  seasonal=seasonal,   # No Seasonality
                                  seasonal_test = "ch",
                                  start_P=1, start_Q=1, start_D=0, method=Methode,
                                  max_P=3, max_Q=3,max_D=2,
                                  trace=True,with_intercept='auto',
                                  error_action='trace',  max_order=max_order,
                                  suppress_warnings=True, return_valid_fits =False,
                                  stepwise=True, approximation=False, n_jobs=1)
                            result=model.fit(Train, ExoTrain[ExoNames])
                            if Kriterium=="aic":
                                Krit.append(result.aic())
                            elif Kriterium=="aicc":
                                Krit.append(result.aicc())
                            elif Kriterium=="bic":
                                Krit.append(result.bic())
                            else:
                                print("Kriterium nicht verfügbar")
                            #Speichern der Ergebnisse in den Listen    
                            Trend_list.append(str(T))
                            # Treiber.append(str(ExoNames))
                            Pred.append(pd.DataFrame(result.predict(Steps,ExoTest[ExoNames]),index=pd.date_range(start=Erster,end=Letzter,freq=Freq)))
                            Fit.append(pd.DataFrame(result.predict_in_sample(ExoTest[ExoNames]),index=Train.index) )
                
              
        #Kriterium "oob" muss gesondert betrachtet werden ->Rest wie oben
        else:
        #Schleife über Treiber Kombinationen
            #Falls Alle werden zudem alle Trends probiert    
              for test in Test:                    
                    for T in Trend:
                        #Spezialfall keine Exogenen Variablen
                        if ExoNames == []:
                                model= auto_arima(Train, start_p=2, start_q=2, start_d=0,
                                      test=test,       # use adftest to find optimal 'd'
                                      max_p=3, max_q=3,max_d=2, # maximum p and q
                                      m=Season,         # frequency of series
                                      # let model determine 'd'
                                      trend=T,out_of_sample_size = 12,
                                      information_criterion=Kriterium,
                                      seasonal=seasonal,   # No Seasonality
                                      seasonal_test = "ch",
                                      start_P=1, start_Q=1, start_D=0, method=Methode,
                                      max_P=3, max_Q=3,max_D=2,
                                      trace=True,with_intercept='auto',
                                      error_action='trace',  max_order=max_order,
                                      suppress_warnings=True, return_valid_fits =False,
                                      stepwise=True, approximation=False,n_jobs=1) 
                                #Speichern des jeweiligen Kriteriums in der Liste
                                result=model.fit(Train)  
                                results.append(result)
                                Krit.append(result.oob())
                                Trend_list.append(str(T))
                                                               
                                Pred.append(pd.DataFrame(result.predict(Steps),index=pd.date_range(start=Erster,end=Letzter,freq=Freq)))
                                Fit.append(pd.DataFrame(result.predict_in_sample(),index=Train.index) )
                        else:
                                model= auto_arima(Train,ExoTrain[ExoNames], start_p=2, start_q=2, start_d=0,
                                      test=test,       # use adftest to find optimal 'd'
                                      max_p=3, max_q=3,max_d=2, # maximum p and q
                                      m=Season,         # frequency of series
                                      # let model determine 'd'
                                      trend=T,out_of_sample_size = 12,
                                      information_criterion=Kriterium,
                                      seasonal=seasonal,   # No Seasonality
                                      seasonal_test = "ch",
                                      start_P=1, start_Q=1, start_D=0, method=Methode,
                                      max_P=3, max_Q=3,max_D=2,
                                      trace=True,with_intercept='auto',
                                      error_action='trace',  max_order=max_order,
                                      suppress_warnings=True, return_valid_fits =False,
                                      stepwise=True, approximation=False, n_jobs=1)
                                result=model.fit(Train,ExoTrain[ExoNames])
                                results.append(result)
                                
                                Krit.append(result.oob())
                                Trend_list.append(str(T))
                                                                
                                Pred.append(pd.DataFrame(result.predict(Steps,ExoTest[ExoNames]),index=pd.date_range(start=Erster,end=Letzter,freq=Freq)))
                                Fit.append(pd.DataFrame(result.predict_in_sample(ExoTrain[ExoNames]),index=Train.index) )
        #Bestes Setup (Trend, Test) zum jeweiligen Treiber Subset Identifizieren und ausgeben 
        Nummer = Krit.index(min(Krit))
        Pred = Pred [Nummer]
        Treiber = ExoNames
        Trend_best = Trend_list[Nummer]
        Fit = Fit[Nummer] 
        
        return Pred, Fit, Trend_best, Treiber, result, Krit
    
    #Führt ein normales SARIMA aus ohne exogene Variablen
    if ExoTrain is None or ExoTest is None:
       ExoNames  = []
       Prediction, Fit_, Trend_best , Treiber, results_best, Krit = Process(ExoNames)
    #Paralellisierung des zuvor definierten Porzesses Satrten (Ein Treibersubset, ein Prozess).      
    else:
        if Optimierung==True:
            if Parallelisierung: 
                Ergebnis = Parallel(n_jobs=-1)(delayed(Process)(ExoNames)  for ExoNames in sub_lists(ExoTrain.columns.tolist()))
            else: 
                Ergebnis = [Process(ExoNames) for ExoNames in sub_lists(ExoTrain.columns.to_list())]
    
        #Overall bestes Modell des paraellelen Prozesses des gewählten Kriteriums extrahieren
            Krit = [item[-1] for item in Ergebnis]
            Nummer = Krit.index(min(Krit))
            #Filtern über alle Model mit kleinstem Kriterium
            results_best = Ergebnis[Nummer][4]
            Trend_best = Ergebnis[Nummer][2]
            Treiber = Ergebnis[Nummer][3]
            Prediction = Ergebnis[Nummer][0]
            Fit_ = Ergebnis[Nummer][1]
        if Optimierung==False: 
            Prediction, Fit_, Trend_best , Treiber, results_best, _ = Process(ExoTrain.columns.to_list())
    print(f'Prediction: {Prediction}, Trend_best: {Trend_best} , Treiber: {Treiber}, results_best: {results_best}')
    return(Prediction, Fit_, Trend_best , Treiber, results_best)
