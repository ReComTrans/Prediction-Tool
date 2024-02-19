import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta   


#fürs Automatisieren der Findung der besten Parameterwerte der Holt-Winters Methode
def exp_smoothing_konfig(seasonal=[None]):
    models = list()
    # definiert die Konfigurationsliste
    t_par = ['add', 'mul', None] #'mul',
    s_par = ['add', 'mul', None]
    d_par = [True, False]
    p_par = seasonal
    b_par = [True, False]
    #r_par = [True, False]
    # erstellt die einzelnen Listeneinträge
    for t in t_par:
        for d in d_par:
            for s in s_par:
                for p in p_par:
                    for b in b_par:
                        #for rb in r_par:
                        kfg = [t,d,s,p,b] #,rb
                        models.append(kfg)
    return models

def HoltWinters(train,step,freq,kfg_list):
    print('Holt Winters Prognose Verfahren startet: \n')
    # Erster=pd.date_range(start=train.index[-1],periods=2, freq=freq)
    if freq=="MS":
        Erster = train.index[-1]+relativedelta(months=1)
        # Letzter =  train.index[-1]+relativedelta(months=step)
    elif freq=="D":
        Erster = train.index[-1]+relativedelta(days=1)
        # Letzter =  train.index[-1]+relativedelta(days=step)
    elif freq=="W":
        Erster = train.index[-1]+relativedelta(weeks=1)
        # Letzter =  train.index[-1]+relativedelta(weeks=step)
    else:
        print("Freq muss MS, D oder W sein")                                             
    params=['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend', 'lamda', 'use_boxcox', 'remove_bias']
    #für Holt Winters
    AIC_list=[]
    #RMSEl=[]
    PARA_list=[]
    MLE_list=[]
    Konfiguration=[]
    Rangpred=pd.DataFrame()
    #df.index.freq = freq
    #je nachdem was unten ausgewählt wurde (RMSE oder AIC) hier dem entsprechend auch auswählen
    #best_RMSE=np.inf
    best_AIC=np.inf
    best_Konfig=[]
    for j in range(len(kfg_list)):
        Konfiguration.append('Mögl. '+str(j+1))
        try:
            kg=kfg_list[j]
            t,d,s,p,bo = kg #,rb
            #damped=True -> gedämpfter Trend
            #smoothing-level (Wert wählbar) -> Glättungslevel (alpha)
            #smoothing-slope (Wert wählbar)-> Glättungssteigung (beta) (für den Trend)
            #smoothing-seasonal(Wert wählbar) -> seasonale Glättung (für die Saison)
            #trend und seasonal können =None, =add oder =mul (Probleme bei mul Aufgrund von negativen Werten)
            #print('Möglichkeit: ' +str(j+1) +' '+ str(kfg_list[j]))
            if t==None:
                model1 = ExponentialSmoothing(train, trend=t, seasonal=s, seasonal_periods=p,initialization_method='estimated', use_boxcox=bo, bounds={'smoothing_level': (0.2,0.8)}, freq=freq)
            else:
                model1 = ExponentialSmoothing(train, trend=t, seasonal=s, damped_trend=d, seasonal_periods=p,initialization_method='estimated', use_boxcox=bo, bounds={'smoothing_level': (0.2,0.8)}, freq=freq)
            #Modell hat teilweise noch Optimierungsprobleme -> werden beim Durchlauf in der Konsole angezeigt
            #"Optimization failed to converge"
            ini = model1.initial_values()
            #print('Anfangswerte: ' + str(ini) +' \n')
            #der Abfang hier ist noch nicht der richtige, die Meldung kommt an anderen Stellen (zu anderen Möglichkeiten)
            #with model1.fix_params({"smoothing_level": 0.2}): #alpha=0.2
            hw_model_fit = model1.fit(optimized=True, method='bh', use_brute=True, minimize_kwargs = {'niter': 100, 'T': 1.0, 'stepsize': 0.5, 'seed': 1}) #remove_bias=rb, minimize_kwargs = {'niter': 1000 , 'T': 10.0, 'stepsize': 1 , 'seed': 1} #,smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma
            
            rangPred = hw_model_fit.forecast(step)
            Rangpred = Rangpred.join(pd.DataFrame(rangPred,columns=['Möglichkeit ' + str(j+1)+' \n'+ str(kfg_list[j])],index=pd.date_range(start=Erster, periods=step, freq=freq)), how='outer')
        
            #rmse = np.sqrt(mean_squared_error(test1[1:],y_Prognose1[1:]))
            AIC=hw_model_fit.aic
            MLE=hw_model_fit.mle_retvals
            PARA=hw_model_fit.params
            AIC_list.append(round(AIC,4))
            MLE_list.append(MLE)
            PARA_list.append([PARA[pa] for pa in params])
            #RMSE_list.append(rmse)
            #je nach Auswahlkriterium AIC oder RMSE muss hier das jeweils andere auskommentiert werden
            #if rmse<best_RMSE:
            if AIC<best_AIC:
                #best_RMSE=rmse
                best_AIC=AIC
                
                best_Konfig=kfg_list[j]
                #der Befehl .copy hatte nicht funktioniert, ohne gibt es das gewünschte Ergebnis aus
                hw_best_model=hw_model_fit
                #Text.write(' \n Parameterangaben der momentan besten Konfiguration(Möglichkeit '+str(j+1) +'): \n' + str(hw_model_fit.params) + ' \n \n')
        except:
            continue
    Rang = list(zip(kfg_list,AIC_list,Konfiguration,PARA_list,MLE_list))
    #Rang_rmse = list(zip(kfg_list,RMSE_list,Konfiguration))
    AIC_list.clear()
    #RMSE_list.clear()
    MLE_list.clear()
    PARA_list.clear()
    Konfiguration.clear()
    #Möglichkeiten ohne AIC entfernen
    Rang = [r for r in Rang if r[1] != None]
    #Rang_rmse = [r for r in Rang_rmse if r[1] != None]
    #sortieren nach AIC-Wert
    Rang.sort(key=lambda Fehler:Fehler[1])
    #Rang_rmse.sort(key=lambda Fehler:Fehler[1])
    try:
        t1,d1,s1,p1,bo1 = best_Konfig #,r1
        #best=str(best_Konfig)
        #hier wird das beste Ergebnis von oben verwendet, ohne diese noch einmal berechnen zu müssen
        hw_model = hw_best_model
    except:
        #wenn use_boxcox wieder mit dazu, dann ist es nicht mehr die 4. Möglichkeit sondern die 1. Möglichkeit (gab vorher einen Fehler aus da bei der 4. saison=mul war)
        #wenn use_boxcox und remove_bias raus dann ist auch die 4. Möglichkeit (add, false, add,12) eine andere als zum Anfang (add,true,add,12,false,false)
        #print('Es wurde die vierte Möglichkeit genutzt')
        #step=12
        #train=y_train['Gesamt']
        best_Konfig=kfg_list[1]#['mul', True, 'mul', 12, True] ; ['add', True, 'mul', 12, True]
        t1,d1,s1,p1,bo1 = best_Konfig #,r1
        
        if t1==None:
            model = ExponentialSmoothing(train, trend=t1, seasonal=s1, seasonal_periods=p1, initialization_method='estimated', use_boxcox=bo1, bounds={'smoothing_level': (0.2,0.8)}, freq=freq)
        else:
            model = ExponentialSmoothing(train, trend=t1, seasonal=s1, damped_trend=d1, seasonal_periods=p1, initialization_method='estimated', use_boxcox=bo1, bounds={'smoothing_level': (0.2,0.8)}, freq=freq)
            #Modell hat teilweise noch Optimierungsprobleme -> werden beim Durchlauf in der Konsole angezeigt
            #"Optimization failed to converge"
        ini = model.initial_values()
        #print('Anfangswerte (kfg_list[1]): '+ str(ini) +' \n')
            #der Abfang hier ist noch nicht der richtige, die Meldung kommt an anderen Stellen (zu anderen Möglichkeiten)
        #with model.fix_params({"smoothing_level": 0.2}): #alpha=0.2
        hw_model = model.fit(optimized=True, method='bh', use_brute=True, minimize_kwargs = {'niter': 100, 'T': 1.0, 'stepsize': 0.5 , 'seed': 1})#remove_bias=r1, minimize_kwargs = {'niter': 1000 , 'T': 10.0, 'stepsize': 1 , 'seed': 1} #method=''Powell, 'L-BFGS-B', 'bh'
        #opti2 = hw_model.optimized
        #damit die Prognose an Trainingsreihe anknüpft und der erste Wert nicht 
        #schon prognostiziert wird: start=test.index[1] 
    para = hw_model.params
    #para['initial_level']=str(para['initial_level'])
    sse = hw_model.sse
    aic = hw_model.aic
    Para=pd.DataFrame(index=['smoothing level alpha','smoothing trend beta','damping trend phi','smoothing seasonal gamma','initial level l_0','initial trend t_0','lambda','boxcox_opti','remove_bias_opti','Trend','Damped','Saison','saisonale Periode','boxcox','SSE','AIC'], columns=['Holt Winters'])
    #r'$\alpha$',r'$\beta$',r'$\phi$',r'$\gamma$','$l_0$','$t_0$',r'$\lambda$'
    Para['Holt Winters'] = [para[p] for p in params] + [best_Konfig[j] for j in range(len(kfg_list[0]))] + [sse] +[aic]
    #print(Para)
    #print(Para.loc['l_0',:])
    #Para.at['l_0','Parameterwerte']=str(Para.at['l_0','Parameterwerte'])
    fit = pd.DataFrame(hw_model.fittedvalues,columns=['Holt Winters'],index=train.index)
    #pred = hw_model.predict(start=test.index[1], end=test.index[-1])
    Pred=hw_model.forecast(step)
    pred = pd.DataFrame(Pred,columns=['Holt Winters'],index=pd.date_range(start=Erster, periods=step, freq=freq))
    print('Beste Konfiguration ist \n')
    print(best_Konfig)
    print('Mit folgenden Parameterwerten: \n')
    print(Para) 
    print('Und folgender Prognose: \n')
    print(pred)
        
    return pred, fit, Para #,best_Konfig,Rangpred
        
