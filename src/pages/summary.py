import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate

import pandas as pd

from dataPreparation.Data import Prediction_pfad as pred_pfad, Parameter_pfad as para_pfad
from methods.methods import methodname


dash.register_page(__name__, path='/Zusammenfassung', name='Zusammenfassung', order = 6) 

layout = html.Div(
    children = [
        html.Div(id = "summary"),
        html.Hr(style={'visibility': 'hidden'}),
        dbc.Button('Reset', id='resetBtn',color='secondary', className='me-1', n_clicks=0),
        dbc.Row([
            dbc.Label(id='Text'),
        ]),   
        html.Div(
            id='downloadContainer',
            hidden=True,
            children=[
                dbc.Row([
                        html.Hr(style={'height':'1px', 
                'margin-bottom':'5px',
                'margin-top':'10px'}),
                ]),
                dbc.Row([
                    dbc.Label('Download der Ergebnisse', style={'fontSize':20, 'textAlign':'center', 'width':'300'}),
                    html.P('Hier können Sie die Prognosen mit Gütekriterien sowie die Parameterwerte/Koeffizienten der Methoden als Datei herunterladen.'),
                    ]),
                dbc.Row([
                    dbc.Col(
                        [
                            dbc.Label('Bitte wählen Sie hier einen Dateinamen:')
                        ],
                        width = 4
                    ),
                    dbc.Col(
                        [
                            dbc.Input(id = 'run_name', placeholder='Dateinamen eingeben ... ', type= 'text')
                        ],
                        # width=4
                        xs=6, sm=6, md=5, lg=5, xl=4, xxl=4
                    )   
                ]),
                html.Hr(style={'visibility': 'hidden'}),
                dbc.Row(
                    [
                        dbc.Col(
                        [
                            dbc.Label('Download als Datei im Excel-Format:'),
                        ],
                        width = 4
                        ),
                        dbc.Col(
                        [
                            dbc.Button('Prognoseergebnisse', id='predictBtnBrowser', color='primary', className='me-1', n_clicks=0, disabled=True),
                            dcc.Download(id='predict_download'),
                        ],
                        width = 5
                        )
                    ],             
                    align='center',
                ),
                html.Div(id='download_predict_browser'),
                dbc.Row([
                    html.Hr(style={'height':'1px', 
                'margin-bottom':'5px',
                'margin-top':'20px'}),
                ]),
                dbc.Row([
                    dbc.Label('Mehr Speicheroptionen', style={'fontSize':20, 'textAlign':'center', 'width':'300'}),
                    html.P('Hier können die Ergebnisse in verschiedenen Formaten im Tool-Ordner abgespeichert werden.'),
                    ]),
                dbc.Row(
                    [dbc.Col([
                            dbc.Label('In welchem Format wollen sie die Tabellen abspeichern?'),
                            dcc.Dropdown(['Excel','CSV'], placeholder='Format...', id='format-selection',multi=True)
                        ],
                        width = 4
                        ),
                    dbc.Col([
                            dbc.Button('Prognoseergebnisse', id='predictBtn', color='primary', className='me-1', n_clicks=0, disabled=True),
                        ],
                        width = 5
                    )],             
                    align='center',
                    ),
                html.Div(id='download_predict'),
            ]
        )
    ]
)

@callback(
    Output('resetBtnStore','data'),
    Output('Text','children'),
    Input('resetBtn','n_clicks')
)
def resetbutton_store(clicks):
    if clicks > 0:
        text = f'Werte wurden {clicks} mal zurück gesetzt!'
    elif clicks is None:
        text = ''
        raise PreventUpdate
    elif clicks == 0:
        text = ''
    store = {
        'data': clicks
    }
    return store, (text)
@callback(
    Output('resetBtn','n_clicks'),
    Input('resetBtnStore','modified_timestamp'),
    State('resetBtnStore', 'data')    
)
def on_resetBtn(timeStamp,data):
    if timeStamp is None:
        raise PreventUpdate
    data = data or {}
    return data.get('data')

@callback(
    Output('summary', 'children'),
    Output('summaryStore','data'),
    State('trainingDateStore', 'data'),
    State('forecastDateStore', 'data'),
    State('forecastingLabelWidthStore', 'data'),
    State('granularityStore', 'data'),
    State('seasonalityStore', 'data'),
    State('productChoice', 'data'),
    State('driverChoice', 'data'),
    State('chosenMethodsStore', 'data'),
    State('unitsStore', 'data'),
    Input('Text','children'),
)
def update_summary(trainDate, forecastDate, forecastingLabelWidth, granularity, seasonality, product, driver, methods, units, text):
    driverData = driver.get('data')
    driverDisplay = ''
    traind = trainDate.get('data')
    if not traind is None:
        traind = pd.to_datetime(traind, format="%d-%m-%Y").strftime('%d.%m.%Y')
    else:
        traind = ''
    forecastd = forecastDate.get('data')
    if not forecastd is None:
        forecastd = pd.to_datetime(forecastd, format="%d-%m-%Y").strftime('%d.%m.%Y')
    else:
        forecastd = ''
    if not seasonality.get('data') is None and not granularity.get('data') is None:
        season = f"{seasonality.get('data')} {granularity.get('data')}"
    else:
        season = ''
    if not forecastingLabelWidth.get('data') is None and not granularity.get('data') is None:
        lof = f"{forecastingLabelWidth.get('data')} {granularity.get('data')}"
    else:
        lof = ''
    if driverData != None:
        for el in driverData:
            driverDisplay += el
            if el != driverData[-1]:
                driverDisplay += ", " 

    methodData = methods.get('data')
    methodsDisplay = ''
    if methodData != None:
        for el in methodData:
            methodsDisplay += el
            if el != methodData[-1]:
                methodsDisplay += ", " 

    children = [
        dbc.Row([
            dbc.Col("Trainingsbeginn", width=2),
            dbc.Col(traind)		
        ]),
        dbc.Row([
            dbc.Col("Prognosebeginn", width=2),
            dbc.Col(forecastd)	
        ]),
        dbc.Row([
            dbc.Col("Granularität", width=2),
            dbc.Col(granularity.get('data'))		
        ]),
        dbc.Row([
            dbc.Col("Zyklus/Saisonalität", width=2),
            dbc.Col(season)	
        ]),
        dbc.Row([
            dbc.Col("Länge Prognosezeitraum", width=2),
            dbc.Col(lof)	
        ]),
        dbc.Row([
            dbc.Col("Produktkategorie", width=2),
            dbc.Col(product.get('data'))	
        ]),
        dbc.Row([
            dbc.Col("Einheit des Produkts", width=2),
            dbc.Col(units.get('data'))	
        ]),
        dbc.Row([
            dbc.Col("Gewählte Treiber", width=2),
            dbc.Col(driverDisplay)
        ]),
        dbc.Row([
            dbc.Col("Gewählte Methode(n)", width=2),
            dbc.Col(methodsDisplay)	
        ]),
    ]
    pre_store = {
        'Produktauswahl' : product.get('data'),
        'Einheit': units.get('data'),
        'Trainingszeitraum' : f'{traind} - {forecastd}',
        'Prognosezeitraum' : f"{forecastingLabelWidth.get('data')} {granularity.get('data')}",
        'Treiberauswahl': driverDisplay,
        'Methodenauswahl': methodsDisplay 
    }
    store = {
        'data': pre_store
    }
    return children, store


@callback(
    Output('runNameStore', 'data', allow_duplicate=True),
    Output('trainingDateStore', 'data', allow_duplicate=True),
    Output('forecastDateStore', 'data',allow_duplicate=True),
    Output('forecastingLabelWidthStore', 'data', allow_duplicate=True),
    Output('granularityStore', 'data', allow_duplicate=True),
    Output('seasonalityStore', 'data', allow_duplicate=True),
    Output('productStore', 'data', allow_duplicate=True),
    Output('productChoice', 'data', allow_duplicate=True),
    Output('driverStore', 'data', allow_duplicate=True),
    Output('driverChoice', 'data', allow_duplicate=True),
    Output('chosenMethodsStore', 'data', allow_duplicate=True),
    Output('unitsStore', 'data', allow_duplicate=True),
    Output(f'{methodname.Holt_Winters}Store','data', allow_duplicate=True),
    Output(f'{methodname.Holt_Winters}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.LASSO}Store','data', allow_duplicate=True),
    Output(f'{methodname.LASSO}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.mlr}Store','data', allow_duplicate=True),
    Output(f'{methodname.mlr}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.ridge}Store','data', allow_duplicate=True),
    Output(f'{methodname.ridge}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.ElasticNet}Store','data', allow_duplicate=True),
    Output(f'{methodname.ElasticNet}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.Sarima}Store','data', allow_duplicate=True),
    Output(f'{methodname.Sarima}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.Sarimax}Store','data', allow_duplicate=True),
    Output(f'{methodname.Sarimax}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.XGB_RF}Store','data', allow_duplicate=True),
    Output(f'{methodname.XGB_RF}ParaStore','data', allow_duplicate=True),
    Output(f'{methodname.GB}Store','data', allow_duplicate=True),
    Output(f'{methodname.GB}ParaStore','data', allow_duplicate=True),
    Output('metricStore','data', allow_duplicate=True),
    Output('exoTestStore','data', allow_duplicate=True),
    Output('SzenarioStore', 'data', allow_duplicate=True),
    Output('variableTrainStore','data', allow_duplicate=True),
    Output('exoTrainStore','data', allow_duplicate=True),
    Output('predictionsStore', 'data', allow_duplicate=True),
    Output('resetBtnStore', 'data', allow_duplicate=True),
    Input('resetBtnStore','data'),
    prevent_initial_call=True
)
def resetStore(resetstore): 
    click = resetstore.get('data')
    if not click is None and click>0:
        return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {'data': 0}
    else:
         raise PreventUpdate
     
#### runName callbacks

@callback(
    Output('run_name', 'value'),
    Input('runNameStore', 'modified_timestamp'),
    State('runNameStore', 'data')
)
def on_runName(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate
    
    data = data or {}

    return data.get('data')

@callback(
    Output('runNameStore', 'data'),
    Output('predictBtnBrowser', 'disabled'),
    Output('run_name', 'invalid'),
    Input('run_name', 'value'),
)
def update_runNameStore(value):
    store = {
        "data": value
    }
    if value is None or value=='':
        return store, True, False
    for c in value: 
        if not (c.isalnum() or c in [' ', '_', '-']):
            return no_update, True, True
    return store, False, False
    
@callback(
    Output('download_predict','children'),
    Output('download_predict_browser','children'),
    Output('predict_download', 'data'),
    Output('predictBtn', 'disabled'),
    Output('downloadContainer', 'hidden'),
    Output('predictBtn','n_clicks'),
    Output('predictBtnBrowser','n_clicks'),
    Input('predictBtn','n_clicks'),
    Input('predictBtnBrowser','n_clicks'),
    Input('format-selection','value'),
    Input('runNameStore', 'data'),
    State('exoTestStore', 'data'),
    State('predictionsStore', 'data'),
    State('metricStore', 'data'),
    State('summaryStore','data'),
    State('chosenMethodsStore', 'data'),
    State(f'{methodname.Holt_Winters}ParaStore','data'),
    State(f'{methodname.LASSO}ParaStore','data'),
    State(f'{methodname.mlr}ParaStore','data'),
    State(f'{methodname.ridge}ParaStore','data'),
    State(f'{methodname.ElasticNet}ParaStore','data'),
    State(f'{methodname.Sarima}ParaStore','data'),
    State(f'{methodname.Sarimax}ParaStore','data'),
    State(f'{methodname.XGB_RF}ParaStore','data'),
    State(f'{methodname.GB}ParaStore','data'),
)
def download_predict(clicks, Bclicks, format, name, driverStore, predStore, metStore, sumStore, 
                     choice, hw_store, lasso_store, mlr_store, ridge_store, net_store, 
                     sarima_store, sarimax_store, XGBRF_store, GB_store):
    if predStore is None or predStore.get('data') is None or len(predStore.get('data'))==0:
        return no_update, no_update, no_update, no_update, True, 0, 0
    predictions = pd.DataFrame.from_dict(predStore.get('data')[0])
    if not metStore is None and not metStore.get('data') is None and metStore.get('data') != []:
        metrics_mape = [pd.DataFrame.from_dict(metStore.get('data')[0][0]),pd.DataFrame.from_dict(metStore.get('data')[0][1])]
        metrics_smape = [pd.DataFrame.from_dict(metStore.get('data')[1][0]),pd.DataFrame.from_dict(metStore.get('data')[1][1])]
        metrics_mae = [pd.DataFrame.from_dict(metStore.get('data')[2][0]),pd.DataFrame.from_dict(metStore.get('data')[2][1])]
        metrics_mfpe = [pd.DataFrame.from_dict(metStore.get('data')[3][0]),pd.DataFrame.from_dict(metStore.get('data')[3][1])]
        metrics_mfe = [pd.DataFrame.from_dict(metStore.get('data')[4][0]),pd.DataFrame.from_dict(metStore.get('data')[4][1])]
    else:
        metrics_mape = pd.DataFrame()
        metrics_smape = pd.DataFrame()
        metrics_mae = pd.DataFrame()
        metrics_mfpe = pd.DataFrame()
        metrics_mfe = pd.DataFrame()
    HW_para = pd.DataFrame.from_dict(hw_store.get('data'))
    regression_para = pd.DataFrame()
    lasso_para = pd.DataFrame.from_dict(lasso_store.get('data'))
    ridge_para = pd.DataFrame.from_dict(ridge_store.get('data'))
    mlr_para = pd.DataFrame.from_dict(mlr_store.get('data'))
    net_para = pd.DataFrame.from_dict(net_store.get('data'))
    sarima_para = pd.DataFrame.from_dict(sarima_store.get('data'))
    sarimax_para = pd.DataFrame.from_dict(sarimax_store.get('data'))
    XGBRF_para = pd.DataFrame.from_dict(XGBRF_store.get('data'))
    GB_para = pd.DataFrame.from_dict(GB_store.get('data'))
    if lasso_store.get('data') != [] and lasso_store != {}:
        regression_para['Koeffizienten'] = lasso_para['Koeffizienten']
        regression_para[methodname.LASSO] = lasso_para[methodname.LASSO]
    if ridge_store.get('data') != [] and ridge_store != {}:
        regression_para['Koeffizienten'] = ridge_para['Koeffizienten']
        regression_para[methodname.ridge] = ridge_para[methodname.ridge]
    if net_store.get('data') != [] and net_store != {}:
        regression_para['Koeffizienten'] = net_para['Koeffizienten']
        regression_para[methodname.ElasticNet] = net_para[methodname.ElasticNet]
    if mlr_store.get('data') != [] and mlr_store != {}:
        if not 'Koeffizienten' in regression_para.columns:
            regression_para['Koeffizienten'] = mlr_para['Koeffizienten']
        regression_para[methodname.mlr] = mlr_para[methodname.mlr]
    choice = choice.get('data')

    if (not format is None and len(format)>0) or Bclicks > 0:
        summaries = pd.DataFrame.from_dict(sumStore.get('data'), orient='index', columns=['Einstellungen'])
        drivers = pd.DataFrame.from_dict(driverStore.get('data'))
        DateiName = f"{name.get('data')}"
        if clicks > 0 or Bclicks > 0:
            if (not format is None and 'Excel' in format) or Bclicks > 0:
                with pd.ExcelWriter(os.path.join(pred_pfad,f'{DateiName}.xlsx'), engine='xlsxwriter') as writer:
                    summaries.to_excel(writer, sheet_name='Einstellungen im Tool', index=True)
                    drivers.to_excel(writer, sheet_name='Testwerte Treiber', index=False)
                    predictions.to_excel(writer, sheet_name='Prognoseergebnisse', index=False)#, float_format="%.2f")
                    metrics_mape[0].to_excel(writer, sheet_name='Gütekriterien - MAPE', index=False)
                    metrics_mape[1].to_excel(writer, sheet_name='Gütekriterien - MAPE', index=False, startrow=len(metrics_mape[0].index)+2)
                    metrics_smape[0].to_excel(writer, sheet_name='Gütekriterien - SMAPE', index=False)
                    metrics_smape[1].to_excel(writer, sheet_name='Gütekriterien - SMAPE', index=False, startrow=len(metrics_smape[0].index)+2)
                    metrics_mae[0].to_excel(writer, sheet_name='Gütekriterien - MAE', index=False)
                    metrics_mae[1].to_excel(writer, sheet_name='Gütekriterien - MAE', index=False, startrow=len(metrics_mae[0].index)+2)
                    metrics_mfpe[0].to_excel(writer, sheet_name='Gütekriterien - proz. Bias', index=False)
                    metrics_mfpe[1].to_excel(writer, sheet_name='Gütekriterien - proz. Bias', index=False, startrow=len(metrics_mfpe[0].index)+2)
                    metrics_mfe[0].to_excel(writer, sheet_name='Gütekriterien - abs. Bias', index=False)
                    metrics_mfe[1].to_excel(writer, sheet_name='Gütekriterien - abs. Bias', index=False, startrow=len(metrics_mfe[0].index)+2)
                    for method in choice:
                        if method == methodname.Holt_Winters:
                            HW_para.to_excel(writer, sheet_name='Parameter'+method, index=True)
                        elif method == methodname.LASSO or method == methodname.mlr or method == methodname.ridge or method == methodname.ElasticNet:
                            regression_para.to_excel(writer, sheet_name='Parameter Regression', index=True)
                        elif method == methodname.Sarima:
                            sarima_para.to_excel(writer, sheet_name='Parameter'+method, index=True)
                        elif method == methodname.Sarimax:
                            sarimax_para.to_excel(writer, sheet_name='Parameter'+method, index=True)
                        elif method == methodname.XGB_RF:
                            XGBRF_para.to_excel(writer, sheet_name='Parameter'+method, index=True)
                        elif method == methodname.GB:
                            GB_para.to_excel(writer,sheet_name='Parameter'+method, index=True)
                    children = [
                        dbc.Label(f'Die Prognosewerte mit den Gütekriterien wurden als Excel in {pred_pfad} unter dem Namen {DateiName} abgespeichert.')
                    ]
            if (clicks > 0 and not format is None and 'CSV' in format):
                summaries.to_csv(os.path.join(pred_pfad,f'{DateiName}_summary.csv'), sep=';', encoding='utf-8', index=True)
                drivers.to_csv(os.path.join(pred_pfad,f'{DateiName}_TestTreiber.csv'), sep=';', encoding='utf-8', index=False)
                predictions.to_csv(os.path.join(pred_pfad,f'{DateiName}_Prognose.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mape[0].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MAPE.csv'), sep=';', encoding='utf-8', index=False)
                metrics_smape[0].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_SMAPE.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mae[0].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MAE.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mfpe[0].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MFPE.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mfe[0].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MFE.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mape[1].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MAPE_separat.csv'), sep=';', encoding='utf-8', index=False)
                metrics_smape[1].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_SMAPE_separat.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mae[1].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MAE_separat.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mfpe[1].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MFPE_separat.csv'), sep=';', encoding='utf-8', index=False)
                metrics_mfe[1].to_csv(os.path.join(pred_pfad,f'{DateiName}_Metrik_MFE_separat.csv'), sep=';', encoding='utf-8', index=False)
                for method in choice:
                    if method == methodname.Holt_Winters:
                        HW_para.to_csv(os.path.join(para_pfad,f'{DateiName}_{method}.csv'), sep=';', encoding='utf-8', index=False)
                    elif method == methodname.LASSO or method == methodname.mlr or method == methodname.ridge or method == methodname.ElasticNet:
                        regression_para.to_csv(os.path.join(para_pfad,f'{DateiName}_Regression.csv'), sep=';', encoding='utf-8', index=False)
                    elif method == methodname.Sarima:
                        sarima_para.to_csv(os.path.join(para_pfad,f'{DateiName}_{method}.csv'), sep=';', encoding='utf-8', index=False)
                    elif method == methodname.Sarimax:
                        sarimax_para.to_csv(os.path.join(para_pfad,f'{DateiName}_{method}.csv'), sep=';', encoding='utf-8', index=False)
                    elif method == methodname.XGB_RF:
                        XGBRF_para.to_csv(os.path.join(para_pfad,f'{DateiName}_{method}.csv'), sep=';', encoding='utf-8', index=False)
                    elif method == methodname.GB:
                        GB_para.to_csv(os.path.join(para_pfad,f'{DateiName}_{method}.csv'), sep=';', encoding='utf-8', index=False)
                children = [
                    dbc.Label(f'Die Prognosewerte mit den Gütekriterien wurden als CSV in {pred_pfad} unter dem Namen {DateiName} abgespeichert.')
                ]
            if (clicks > 0 and not format is None and ['Excel', 'CSV'] in format):
                children = [
                    dbc.Label(f'Die Prognosewerte mit den Gütekriterien wurden als Excel und CSV in {pred_pfad} unter dem Namen {DateiName} abgespeichert.')
                ]
            if not format is None and len(format)>0:
                button_switch = False
            if clicks == 0: 
                children = []
            if Bclicks == 0: 
                Bchildren = []
                download = no_update
            if Bclicks > 0:
                download = dcc.send_file(os.path.join(pred_pfad,f'{DateiName}.xlsx'))
                os.remove(os.path.join(pred_pfad,f'{DateiName}.xlsx'))
                Bchildren = [
                    dbc.Label('Download der Prognosewerte mit den Gütekriterien als Excel sollte in Kürze starten.')
                ]
                if format is None or len(format)==0:
                    button_switch = True
        else:
            if DateiName is None or DateiName=='': 
                button_switch = True
            else:
                button_switch = False
            children = []
            Bchildren = []
            download = no_update
    else:
        button_switch = True
        children = []
        Bchildren = []
        download = no_update
    return children, Bchildren, download, button_switch, False, 0, 0

