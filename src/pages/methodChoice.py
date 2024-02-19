import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import dcc, html, callback, dash_table, Output, Input, State, DiskcacheManager, no_update
from dash.exceptions import PreventUpdate

# -------- for plots -----------------
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# ----------- 
import pandas as pd
import numpy as np
import sys, os, signal
import time
# ----------- import own source code -----------
from metrics.Fehleranalyse import (
    metric_MAPE, metric_MAPE_pj, metric_MAPE_separate, metric_MAE, metric_MAE_pj, metric_MAE_separate,
    metric_SMAPE, metric_SMAPE_pj, metric_SMAPE_separate, metric_MFE, metric_total_bias, 
    metric_rel_total_bias, metric_bias_separate, metric_rel_bias_separate, metric_rel_MFE
    )

from methods.holt_winters import exp_smoothing_konfig, HoltWinters
from methods.Lasso import LASSO
from methods.lineareRegression import MLR
from methods.Ridge import Ridge
from methods.ElasticNet import ElasticNet
# from methods.auto_arima import AutoSARIMA
# from methods.auto_sarimax import AutoSARIMAX
from methods.auto_sarimax_variable import AutoSARIMAX
from methods.RandomForest import XGB_RandomForest
from methods.GradientBoosting import XGBoost
from methods.methods import methodname
dash.register_page(__name__, path='/Methodenauswahl', name='Methodenauswahl', order = 5)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# _running_calculations = {n: False for n in methodname.List} 

options_list = [
    {'label': html.Span([methodname.mlr], style={'color': 'Blue'}), 'value': methodname.mlr},
    {'label': html.Span([methodname.LASSO], style={'color': 'Blue'}), 'value': methodname.LASSO},
    {'label': html.Span([methodname.ridge], style={'color': 'Blue'}), 'value': methodname.ridge},
    {'label': html.Span([methodname.ElasticNet], style={'color': 'Blue'}), 'value': methodname.ElasticNet},
    {'label': html.Span([methodname.Holt_Winters], style={'color': 'Goldenrod'}), 'value': methodname.Holt_Winters},
    {'label': html.Span([methodname.Sarima], style={'color': 'Goldenrod'}), 'value': methodname.Sarima},
    {'label': html.Span([methodname.Sarimax], style={'color': 'Green'}), 'value': methodname.Sarimax},
    {'label': html.Span([methodname.XGB_RF], style={'color': 'Darkorchid'}), 'value': methodname.XGB_RF},
    {'label': html.Span([methodname.GB], style={'color': 'Darkorchid'}), 'value': methodname.GB},
]

layout = html.Div(
	children = [
        dbc.Row(
                [dbc.Col(
                    [
                        dbc.Label('Welche(s) Prognoseverfahren möchten Sie wählen?')
                    ],
                    width = 3
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(id="method-selection", placeholder="Prognoseverfahren wählen...", options=options_list, multi=True),
                    ],
                    width = 5
                )],             
            
            align='center',           
        ),
        dbc.Row([
        html.Label([
            "Prognose basierend auf: ",
            html.Span(['Nur Treibern'], style={'color': 'Blue'}), ', ', 
            html.Span(['Nur Produktdaten'], style={'color': 'Goldenrod'}), ' und ', 
            html.Span(['Treibern und Produktdaten'], style={'color': 'Green'}),'.']),
        ]),
        dbc.Row([
        html.Label([
            "Prognose basierend auf: ", 
            html.Span(['Nur Treibern'], style={'color': 'Darkorchid'}), 
            ", aber mit reduzierter Erklärbarkeit."
        ])
        ]),
        html.P('Mehr Informationen in der Dokumentation.'),
        html.Hr(style={'height':'1px', 
                       'visibility':'hidden', 
                       'margin-bottom':'-3px',}),
        dbc.Row(
            dbc.Switch(id= 'display-table', label='Szenarien und Eigene Prognosen eingeben/ändern'),
        ),
        dbc.Row(
            html.Div(id='Tabellen',
            children =
                [
                dbc.Label('Tabelle der Testwerte für die ausgewählten Treiber und Tabelle für eigenen Prognosen',color='secondary'),
                dbc.Row([
                    dbc.Col(
                        [ 
                            dbc.Label(id='Label_ExoTest'),
                            dash_table.DataTable(id='ExoTest', 
                                                 editable = True,
                                                 style_cell={
                                                     'textAlign': 'left',
                                                     'whiteSpace': 'normal',
                                                     'height': 'auto',
                                                     'width': 'auto'},
                                                 fixed_columns={'headers': True, 'data': 1},
                                                 style_table={'overflowX': 'auto'},
                                                 page_size=12,)
                        ],width=7),
                    dbc.Col(
                        [
                            dbc.Label(id='Label_Szenario'),
                            dash_table.DataTable(id='Szenario',
                                                 editable = True,
                                                 fixed_columns = {'headers': True, 'data': 1},
                                                 style_cell={'textAlign': 'left',
                                                             'whiteSpace': 'normal',
                                                             'height': 'auto',},
                                                  page_size=12)
                        ], width = 4),
                ]),
                html.Hr(style={'height':'1px', 
                       'visibility':'hidden', 
                       'margin-bottom':'-3px',}),
                dbc.Row([
                    dbc.Col([
                            dbc.Button('Tabellenänderung speichern',id='change-button', color='success', className='me-1', n_clicks=0),
                        ], width= 4),
                    dbc.Col([
                            dbc.Input(id='editing-prediction',
                                      placeholder='Spaltenname Ihrer Prognose...',
                                      value='',
                                      style={'padding': 7}
                                      ),
                        ], width= 4),
                    dbc.Col([
                            dbc.Button('Spalte hinzufügen', id='editing-prediction-button', color='primary', className='me-1', n_clicks=0),
                    ], width=4),
                ]),
                ], style= {'display' : 'none'}
            ),
        ),
        html.Hr(),
        dbc.Row(
            dbc.Col([
                html.Div([
                    dbc.Button('Prognose Starten', id='prediction-button', color='primary', className='me-1', n_clicks=0, disabled=False),
                    dbc.Button('Tool stoppen', id='cancel-button', color='secondary', className='me-1', n_clicks=0, disabled=True),
                    dbc.Label('', id='cancel_button_reminder')
                ]),
            ]),
        ),
        html.Div(html.P(id='initialText', children=[])),
        html.Hr(),
        dbc.Row([
                dbc.Col([
                   dbc.Button('Ergebnisse anzeigen', id='results-button', color='primary', className='me-1', n_clicks=0),
                ], width=3), 
                dbc.Col([
                    dbc.RadioItems(
                        options={'pred': 'Prognose','metr': 'Metriken', 'fit': 'Trainingsgüte', 'para': 'Parameterwerte/Treiberbewertung'}, 
                        id='display-selection', inline=True
                        )
                ])
                ]),
        html.Hr(style={'height':'1px', 
                       'visibility':'hidden', 
                       'margin-bottom':'-3px',}),
        dbc.Row(
            [
                html.Div(id='hw_message', children=[]),
                html.Div(id='lasso_message', children=[]),
                html.Div(id='mlr_message', children=[]),
                html.Div(id='ridge_message', children=[]),
                html.Div(id='net_message', children=[]),
                html.Div(id='sarima_message', children=[]),
                html.Div(id='sarimax_message', children=[]),
                html.Div(id='xgbrf_message', children=[]),
                html.Div(id='gb_message', children=[]),
            ]
        ),
        html.Div([
            html.Div(id='Grafik'),
            html.Hr(style={'height':'1px', 
                        'visibility':'hidden', 
                        'margin-bottom':'-3px',}),
            dbc.Row([
                dbc.Col([html.Div(dash_table.DataTable(id='PredictionTable',
                                                    page_size=12,
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto'
                                                    },
                                                    #fill_width=False,
                                        )),],id='PredTabWidth', width=3),
            ]),
        ], id='predContainer', hidden=True),
        html.Div([
            dbc.RadioItems({'MAPE':'MAPE (Prozentual)', 'SMAPE':'SMAPE (Prozentual)', 
                            'MAE':'MAE (Absolut)', 'MFPE':'Bias (Prozentual)', 'MFE':'Bias (Absolut)'}, 
                           id='metric-display-selection', inline=True, value='MAPE'),
            html.Div(id='MetrikMAPE', hidden=True),
            html.Div(id='MetrikMAE', hidden=True),
            html.Div(id='MetrikSMAPE', hidden=True),
            html.Div(id='MetrikMFPE', hidden=True),
            html.Div(id='MetrikMFE', hidden=True),
        ], id='metrikContainer', hidden=True),
        html.Div([
        html.Div(id='Fit_Grafik'),
        html.Div(id='Fit_Metrik'),
        ], id='fitContainer', hidden=True),
        html.Div([
            dbc.RadioItems([], id='param-display-selection', inline=True),
            html.Hr(style={'visibility':'hidden'}),
            html.Div(id=methodname.Holt_Winters, children=[], hidden=True),
            html.Div(id=methodname.LASSO, children=[], hidden=True),
            html.Div(id=methodname.mlr, children=[], hidden=True),
            html.Div(id=methodname.ridge, children=[], hidden=True),
            html.Div(id=methodname.ElasticNet, children=[], hidden=True),
            html.Div(id=methodname.Sarima, children=[], hidden=True),
            html.Div(id=methodname.Sarimax, children=[], hidden=True),
            html.Div(id=methodname.XGB_RF, children=[], hidden=True),
            html.Div(id=methodname.GB, children=[], hidden=True),
        ], id='paramContainer', hidden=True),
        html.Hr(style={'height':'1px', 
                       'visibility':'hidden', 
                       'margin-bottom':'-3px',}),    ]
)

@callback(
    Output('Label_Szenario','children'),
    Output('Szenario','page_size'),
    Output('Label_ExoTest', 'children'),
    Output('ExoTest','page_size'),
    Input('granularityStore','data'),
    Input('forecastingLabelWidthStore', 'data'), 
)
def add_granularity(gran, lof):
    granularity = gran.get('data')
    gran_num = 7 if granularity=='Tage' else 12 # ggf noch anpassen für andere Saisonalität
    return (
            f'Eigene Prognosen in {granularity}: ',
            min(lof.get('data'), gran_num),
            f'Treiber für den Prognosezeitraum in {granularity}: ',
            min(lof.get('data'), gran_num)
            )

@callback(
    Output('ExoTest', 'data'),
    Output('Szenario', 'data'),
    Output('variableTrainStore', 'data'),
    Output('exoTrainStore', 'data'),
    Output('exoTestStore', 'data'),
    Output('Tabellen', 'style'),
    Output('ExoTest', 'columns'),
    Output('SzenarioStore', 'data'),
    Output('Szenario', 'columns'),
    Input('display-table', 'value'),
    State('driverStore', 'data'),
    State('driverChoice', 'data'),
    State('productStore', 'data'),
    State('productChoice', 'data'), 
    State('trainingDateStore', 'data'),
    State('forecastDateStore', 'data'),
    State('forecastingLabelWidthStore', 'data'),
    State('granularityStore', 'data'),
    State('exoTestStore', 'data'),
    State('SzenarioStore', 'data'),
    State('exoTestStore', 'modified_timestamp'),
    State('driverChoice', 'modified_timestamp'), 
    State('driverStore', 'modified_timestamp'),
    State('trainingDateStore', 'modified_timestamp'),
    State('forecastDateStore', 'modified_timestamp'),
    State('forecastingLabelWidthStore', 'modified_timestamp'),
    State('granularityStore', 'modified_timestamp'),
    Input('prediction-button', 'n_clicks'),
)
def data_split(display,driverStore,driverc,productStore,productc,Datem,Datec,lof,gran,driverTest,
               szenarioStore, time_exo, time_driverc, time_drivers, time_td, time_fd, time_flen, 
               time_gran, _):
    Anzeige = '%d-%m-%Y'
    driver_choice = driverc.get('data')
    driver = pd.DataFrame.from_dict(driverStore.get('data'))
    driver.set_index('Date', drop=False, inplace=True)
    driver.index = pd.to_datetime(driver.index, format=Anzeige)
    driver[driver.columns[1:]] = driver[driver.columns[1:]].astype(float)
    product_df = pd.DataFrame.from_dict(productStore.get('data'))
    product_df.set_index('Date', inplace=True)
    product_df.index = pd.to_datetime(product_df.index, format=Anzeige)
    product_df = product_df.astype(float)
    product_choice = productc.get('data')
    Date_min = Datem.get('data')
    Date_cut = Datec.get('data')
    lenght_of_forecast = lof.get('data')
#Trainings und Testdaten erstellen
    y_train = pd.DataFrame(columns=['Date'])
    exo_train = pd.DataFrame(columns=['Date'])
    exo_test = pd.DataFrame(columns=['Date'])
    szenario = pd.DataFrame(columns=['Date'])
    y_train[product_choice] = product_df[product_choice][((pd.to_datetime(product_df.index, format=Anzeige) >= pd.to_datetime(Date_min, format=Anzeige))&
                                (pd.to_datetime(product_df.index, format=Anzeige) < pd.to_datetime(Date_cut, format=Anzeige)))]
    y_train['Date'] = pd.to_datetime(y_train.index, format=Anzeige)
    y_train['Date'] = y_train['Date'].dt.strftime(Anzeige)
    for c in driver_choice:
        exo_train[c] = driver[c][((pd.to_datetime(driver.index, format=Anzeige) >= pd.to_datetime(Date_min, format=Anzeige))&
                                (pd.to_datetime(driver.index, format=Anzeige) < pd.to_datetime(Date_cut, format=Anzeige)))]
        exo_train['Date'] = pd.to_datetime(exo_train.index, format=Anzeige)
        exo_train['Date'] = exo_train['Date'].dt.strftime(Anzeige)
    if (
        driverTest is None or 
        driverTest == {} or 
        driverTest.get('data') == [] or
        time_exo < time_driverc or
        time_exo < time_drivers or
        time_exo < time_td or
        time_exo < time_fd or
        time_exo < time_flen or
        time_exo < time_gran
        ):
        for c in driver_choice:
            exo_test[c] = driver[c][(pd.to_datetime(driver.index, format=Anzeige) >= pd.to_datetime(Date_cut, format=Anzeige))]
        exo_test = exo_test[driver_choice][0:lenght_of_forecast]
        len_test = len(exo_test.index)
        if len_test < lenght_of_forecast:
            test_dates = exo_test.index.tolist()
            if len_test > 0:
                last = pd.to_datetime(exo_test.index[-1], format=Anzeige) 
            else:
                last = pd.to_datetime(driver.index[-1], format=Anzeige) 
            if gran['data'] == 'Tage':  
                last += pd.Timedelta(1, 'D')
            elif gran['data'] == 'Monate':
                if last.month in [1,3,5,7,8,10,12]:
                    last += pd.Timedelta(31, 'D')
                elif last.month in [4,6,9,11]:
                    last += pd.Timedelta(30, 'D')
                else: 
                    last += pd.Timedelta(28, 'D') if last.year%4 else pd.Timedelta(29, 'D')
            test_dates.append(last)
            if lenght_of_forecast - len_test > 1:
                for _ in range(lenght_of_forecast - len_test - 1):
                    last = test_dates[-1]
                    if gran['data'] == 'Tage':  
                        last += pd.Timedelta(1, 'D')
                    elif gran['data'] == 'Monate':
                        if last.month in [1,3,5,7,8,10,12]:
                            last += pd.Timedelta(31, 'D')
                        elif last.month in [4,6,9,11]:
                            last += pd.Timedelta(30, 'D')
                        else: 
                            last += pd.Timedelta(28, 'D') if last.year%4 else pd.Timedelta(29, 'D')
                    test_dates.append(last)
            exo_test = pd.DataFrame(exo_test, index=test_dates)
            exo_test.insert(0,'Date', exo_test.index)
        else:
            exo_test.insert(0,'Date', exo_test.index)
        exo_test['Date'] = pd.to_datetime(exo_test['Date'], format=Anzeige).dt.strftime(Anzeige)
        szenario = pd.DataFrame(exo_test['Date'])
    else:
        exo_test = pd.DataFrame.from_dict(driverTest.get('data'))
        if not szenarioStore is None and szenarioStore != {}:
            szenario = pd.DataFrame.from_dict(szenarioStore.get('data'))
        else:
            szenario['Date'] = pd.DataFrame(exo_test['Date'])
    variableTrainStore = {
        'data' : y_train.to_dict('records')
    }
    driverTrainStore = {
        'data' : exo_train.to_dict('records')
    }
    driverTestStore = {
        'data' : exo_test.to_dict('records')
    }
    SzenarioStore ={
        'data' : szenario.to_dict('records')
    }
    columns = [{'name' : i, 'id' : i} for i in exo_train.columns]
    columns[0]['editable'] = False
    columns_szenario = [{'name' : i, 'id' : i, 'renamable': True, 'deletable': True} 
                        if i != 'Date' else {'name' : i, 'id' : i, 'editable': False} for i in szenario.columns]
    if display:
        return (driverTestStore.get('data'), SzenarioStore.get('data'),
                variableTrainStore, driverTrainStore, driverTestStore, 
                {'display' : 'inline'}, columns, SzenarioStore, columns_szenario)
    else:
        return (no_update, no_update, 
                variableTrainStore, driverTrainStore, driverTestStore, 
                {'display' : 'none'}, no_update, SzenarioStore, no_update)

@callback(
    Output('Szenario', 'columns', allow_duplicate=True),
    Output('editing-prediction-button', 'n_clicks'),
    Input('editing-prediction-button', 'n_clicks'),
    State('editing-prediction', 'value'),
    State('Szenario', 'columns'),
    prevent_initial_call=True
)
def update_columns(n_clicks, value, existing_columns):
    if n_clicks > 0:
        existing_columns.append({
            'id': value, 'name': value,
            'renamable': True, 'deletable': True,
        })
        return existing_columns, 0
    else:
        raise PreventUpdate

@callback(
    Output('exoTestStore','data', allow_duplicate=True),
    Output('SzenarioStore', 'data', allow_duplicate=True),
    Output('change-button','n_clicks'),
    Output('initialText','children', allow_duplicate=True),
    Output('initialText','style', allow_duplicate=True),
    Input('change-button','n_clicks'),
    State('ExoTest', 'data'),
    State('Szenario', 'data'),
    State('Szenario', 'columns'),
    prevent_initial_call=True
)
def update_exoTestStore(clicks,exoTest,szenario,columns):
    if clicks > 0:
        try: 
            exoTest = pd.DataFrame.from_dict(exoTest)
            exoTest[exoTest.columns[1:]] = exoTest[exoTest.columns[1:]].astype(float)
        except: 
            children = 'Fehlerhafte Treiberdaten.'
            style = {'color': 'tomato'}
            return no_update, no_update, 0, children, style
        store = {
            'data': exoTest.to_dict('records')
        }
        children = 'Szenario-Tabelle gespeichert.'
        style = {'color': 'green'}
        col_names = [c['name'] for c in columns]
        if '' in col_names: 
            children = 'Bitte Spalte benennen'
            szenariostore = {}
            style = {'color': 'tomato'}
            return store, szenariostore, 0, children, style
        df = pd.DataFrame.from_records(szenario)
        if len(df.columns) < len(col_names):
            children = 'Eine Spalte ist leer'
            szenariostore = {}
            style = {'color': 'tomato'}
            return store, szenariostore, 0, children, style
        df.columns = col_names
        szenariostore = {
            'data': df.to_dict('records')
        }
        for c in col_names: 
            if (df[c].isna()).any() or (df[c]==0).any() or (df[c]=='').any(): 
                children = f'Prognosewerte sind nicht vollständig in {c}'
                szenariostore = {}
                style = {'color': 'red'}
        return store, szenariostore, 0, children, style
    else:
        return no_update, no_update, no_update, no_update, no_update

#### methods callbacks
@callback(
    Output('chosenMethodsStore', 'data'),
    Output('param-display-selection', 'options'),
    Output('param-display-selection', 'value'),
    Input('method-selection', 'value')
)
def update_method_selection(choice):
    store = {
        "data": choice
    }
    select = choice[0] if choice else []
    return store, choice if choice else [], select

@callback(
    Output('method-selection', 'value'),
    Input('chosenMethodsStore', 'modified_timestamp'),
    State('chosenMethodsStore', 'data')
)
def on_method(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate 
    data = data or {}
    return data.get('data')

@callback(
    Output('initialText','children', allow_duplicate=True),
    Output('initialText','style', allow_duplicate=True),
    Output('prediction-button', 'disabled', allow_duplicate=True),
    Output('results-button', 'disabled', allow_duplicate=True),
    Output("Uploader", "disabled", allow_duplicate=True),
    Output("Voreinstellungen", "disabled", allow_duplicate=True),
    Output("Methodenauswahl", "disabled", allow_duplicate=True), 
    Output("Produktauswahl", "disabled", allow_duplicate=True), 
    Output("Treiberauswahl", "disabled", allow_duplicate=True),
    Output("Zusammenfassung", "disabled", allow_duplicate=True),
    Output('cancel-button', 'disabled', allow_duplicate=True),
    Output('cancel-button', 'color', allow_duplicate=True),
    Output('cancel_button_reminder', 'children', allow_duplicate=True),
    Output('display-table','value', allow_duplicate=True),
    Output('display-selection', 'options', allow_duplicate=True),
    Output('display-selection', 'value', allow_duplicate=True),
    Output('hw_message', 'children', allow_duplicate=True),
    Output('lasso_message', 'children', allow_duplicate=True),
    Output('mlr_message', 'children', allow_duplicate=True),
    Output('ridge_message', 'children', allow_duplicate=True),
    Output('net_message', 'children', allow_duplicate=True),
    Output('sarima_message', 'children', allow_duplicate=True),
    Output('sarimax_message', 'children', allow_duplicate=True),
    Output('xgbrf_message', 'children', allow_duplicate=True),
    Output('gb_message', 'children', allow_duplicate=True),
    Output('predContainer', 'hidden', allow_duplicate=True),
    Output('metrikContainer', 'hidden', allow_duplicate=True),
    Output('fitContainer', 'hidden', allow_duplicate=True),
    Output('paramContainer', 'hidden', allow_duplicate=True),
    Input('prediction-button', 'n_clicks'),
    State('method-selection', 'value'),
    State('display-table','value'),
    State('exoTestStore', 'data'),
    prevent_initial_call=True
)
#children könnte später vielleicht auch die progressbar werden?
def method_initial(clicks,choice,display,exoTest):
    style = {'color': ''}
    print(f'display Beginn: {display}')
    if choice is None or choice == []: 
        raise PreventUpdate
    if clicks > 0 and not exoTest is None and not exoTest.get('data') is None:
        d_opt = [
            {'value':'pred','label':'Prognose','disabled':True},
            {'value':'metr','label':'Metriken','disabled':True},
            {'value':'fit','label':'Trainingsgüte','disabled':True},
            {'value':'para','label':'Parameterwerte/Treiberbewertung','disabled':True},
        ]
        d_val = None
        d = True
        childrenIni = 'Methoden starten'
        display = False
        ccolor = 'danger'
        cancel_text = '(Abbruch erfordert einen Neustart des Tools)'
        exo_df = pd.DataFrame.from_dict(exoTest.get('data'))
        if [d for d in exo_df.columns if exo_df[d].isna().sum()>0]:
            d_opt = no_update
            d_val = no_update
            d = False
            ccolor = no_update
            cancel_text = no_update
            display = True
            childrenIni = html.Span(
                ['Nicht genug Treiberdaten im Prognosezeitraum vorhanden! Bitte oben nachtragen.'], 
                style={'color': 'tomato'}
                                    )
    else:
        d_opt = no_update
        d_val = no_update
        d = False
        ccolor = no_update
        cancel_text = no_update
        childrenIni = no_update 
    print(f'display Ende: {display}')
    msg_HW = dbc.Label(f'Methode {methodname.Holt_Winters} rechnet noch.', style={'color':'tomato'})
    msg_L = dbc.Label(f'Methode {methodname.LASSO} rechnet noch.', style={'color':'tomato'})
    msg_M = dbc.Label(f'Methode {methodname.mlr} rechnet noch.', style={'color':'tomato'})
    msg_R = dbc.Label(f'Methode {methodname.ridge} rechnet noch.', style={'color':'tomato'})
    msg_EN = dbc.Label(f'Methode {methodname.ElasticNet} rechnet noch.', style={'color':'tomato'})
    msg_S = dbc.Label(f'Methode {methodname.Sarima} rechnet noch.', style={'color':'tomato'})
    msg_SX = dbc.Label(f'Methode {methodname.Sarimax} rechnet noch.', style={'color':'tomato'})
    msg_XGBRF = dbc.Label(f'Methode {methodname.XGB_RF} rechnet noch.', style={'color':'tomato'})
    msg_GB = dbc.Label(f'Methode {methodname.GB} rechnet noch.', style={'color':'tomato'})

    return (childrenIni, style, d, d, d, d, d, d, d, d, not d, ccolor, cancel_text, display, d_opt, d_val, 
            msg_HW, msg_L, msg_M, msg_R, msg_EN, msg_S, msg_SX, msg_XGBRF, msg_GB, True, True, True, True)

#----------------- methods callbacks -------------------
@callback(
    Output(f'{methodname.Holt_Winters}Store','data'),
    Output(f'{methodname.Holt_Winters}ParaStore','data'),
    Output('hw_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('forecastingLabelWidthStore','data'),
    State('unitsStore', 'data'),
    State('granularityStore', 'data'),
    State('seasonalityStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_HW(clicks, choice, ytrain, lof, unit, gran, season, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
        
    # _running_calculations[methodname.Holt_Winters] = True

    hw_Pred = pd.DataFrame(columns=['Date',methodname.Holt_Winters])
    hw_Fit = pd.DataFrame(columns=['Date',methodname.Holt_Winters])
    HW_para = pd.DataFrame()
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        lenght_of_forecast = lof.get('data')
        if methodname.Holt_Winters in choice:
            seasonality = season.get('data')
            granularity = gran.get('data')
            freq = 'D' if granularity=='Tage' else 'MS'
            kfgl = exp_smoothing_konfig(seasonal=[seasonality])# [0,4,6,12]
            HW_pred, HW_fit, HW_para = HoltWinters(y_train, int(lenght_of_forecast), freq, kfgl)
            hw_Pred[methodname.Holt_Winters] = HW_pred[methodname.Holt_Winters].round(Round)
            hw_Pred['Date'] = HW_pred.index
            hw_Pred['Date'] = hw_Pred['Date'].dt.strftime(Anzeige)
            hw_Fit[methodname.Holt_Winters] = HW_fit[methodname.Holt_Winters].round(Round)
            hw_Fit['Date'] = HW_fit.index
            hw_Fit['Date'] = hw_Fit['Date'].dt.strftime(Anzeige)
            HW_para.insert(0, 'Parameter', HW_para.index.astype(str))
            hw_Pred.replace(0, 0.0, inplace=True)
            hw_Fit.replace(0, 0.0, inplace=True)
            HW_para.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.Holt_Winters} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.Holt_Winters} wurde nicht ausgewählt.', style={'color':'grey'})
    else:
        print(f'Drücken Sie den Knopf (Holt Winters clicks = {clicks})')
        msg = dbc.Label('Drücken Sie den Knopf (Holt Winters)', style={'color':'grey'})
    store = {
        'data': [hw_Pred.to_dict('records'), hw_Fit.to_dict('records')]
    }
    para_store = {
        'data': HW_para.to_dict('records')
    }
    # _running_calculations[methodname.Holt_Winters] = False
    return store, para_store, msg
    
@callback(
    Output(f'{methodname.LASSO}Store','data'),
    Output(f'{methodname.LASSO}ParaStore','data'),
    Output('lasso_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_LASSO(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    
    # _running_calculations[methodname.LASSO] = True

    lasso_Pred = pd.DataFrame(columns=['Date',methodname.LASSO])
    lasso_Fit = pd.DataFrame(columns=['Date',methodname.LASSO])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.LASSO in choice:
##############################LASSO-Methode    
            lasso_pred, lasso_fit, lasso_koeff = LASSO(y_train, exo_train, exo_test)
            lasso_Pred[methodname.LASSO] = lasso_pred.round(Round)
            lasso_Pred['Date'] = lasso_pred.index
            lasso_Pred['Date'] = lasso_Pred['Date'].dt.strftime(Anzeige)
            lasso_Pred = lasso_Pred[['Date',methodname.LASSO]]
            lasso_Fit[methodname.LASSO] = lasso_fit.round(Round)
            lasso_Fit['Date'] = lasso_fit.index
            lasso_Fit['Date'] = lasso_Fit['Date'].dt.strftime(Anzeige)
            lasso_Fit = lasso_Fit[['Date',methodname.LASSO]]
            lasso_koeff = lasso_koeff.round(Round)
            lasso_koeff.insert(0,'Koeffizienten',lasso_koeff.index.astype(str))
            lasso_Pred.replace(0, 0.0, inplace=True)
            lasso_Fit.replace(0, 0.0, inplace=True)
            lasso_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.LASSO} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.LASSO} wurde nicht ausgewählt.', style={'color':'grey'})   
            lasso_koeff = pd.DataFrame(columns=['Koeffizienten','LASSO'])
    else:
        lasso_koeff = pd.DataFrame(columns=['Koeffizienten','LASSO'])
        print(f'Drücken Sie den Knopf (LASSO clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (LASSO clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [lasso_Pred.to_dict('records'), lasso_Fit.to_dict('records')]
    } 
    para_store = {
        'data': lasso_koeff.to_dict('records')
    }
    print(f'lasso returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.LASSO] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.mlr}Store','data'),
    Output(f'{methodname.mlr}ParaStore','data'),
    Output('mlr_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_MLR(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    
    # _running_calculations[methodname.mlr] = True

    mlr_Pred = pd.DataFrame(columns=['Date',methodname.mlr])
    mlr_Fit = pd.DataFrame(columns=['Date',methodname.mlr])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.mlr in choice:
##############################MLR-Methode    
            mlr_pred, mlr_fit, mlr_koeff = MLR(y_train, exo_train, exo_test)
            mlr_Pred[methodname.mlr] = mlr_pred.round(Round)
            mlr_Pred['Date'] = mlr_pred.index
            mlr_Pred['Date'] = mlr_Pred['Date'].dt.strftime(Anzeige)
            mlr_Pred = mlr_Pred[['Date',methodname.mlr]]
            mlr_Fit[methodname.mlr] = mlr_fit.round(Round)
            mlr_Fit['Date'] = mlr_fit.index
            mlr_Fit['Date'] = mlr_Fit['Date'].dt.strftime(Anzeige)
            mlr_Fit = mlr_Fit[['Date',methodname.mlr]]
            mlr_koeff = mlr_koeff.round(Round)
            mlr_koeff.insert(0,'Koeffizienten',mlr_koeff.index.astype(str))
            mlr_Pred.replace(0, 0.0, inplace=True)
            mlr_Fit.replace(0, 0.0, inplace=True)
            mlr_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.mlr} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.mlr} wurde nicht ausgewählt.', style={'color':'grey'})
            mlr_koeff = pd.DataFrame(columns=['Koeffizienten','MLR'])
    else:
        mlr_koeff = pd.DataFrame(columns=['Koeffizienten','MLR'])
        print(f'Drücken Sie den Knopf (MLR clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (MLR clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [mlr_Pred.to_dict('records'), mlr_Fit.to_dict('records')]
    } 
    para_store = {
        'data': mlr_koeff.to_dict('records')
    }
    print(f'mlr returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.mlr] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.ridge}Store','data'),
    Output(f'{methodname.ridge}ParaStore','data'),
    Output('ridge_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_Ridge(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    
    # _running_calculations[methodname.ridge] = True

    ridge_Pred = pd.DataFrame(columns=['Date',methodname.ridge])
    ridge_Fit = pd.DataFrame(columns=['Date',methodname.ridge])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.ridge in choice:
##############################Ridge-Methode    
            ridge_pred, ridge_fit, ridge_koeff = Ridge(y_train, exo_train, exo_test)
            ridge_Pred[methodname.ridge] = ridge_pred.round(Round)
            ridge_Pred['Date'] = ridge_pred.index
            ridge_Pred['Date'] = ridge_Pred['Date'].dt.strftime(Anzeige)
            ridge_Pred = ridge_Pred[['Date',methodname.ridge]]
            ridge_Fit[methodname.ridge] = ridge_fit.round(Round)
            ridge_Fit['Date'] = ridge_fit.index
            ridge_Fit['Date'] = ridge_Fit['Date'].dt.strftime(Anzeige)
            ridge_Fit = ridge_Fit[['Date',methodname.ridge]]
            ridge_koeff = ridge_koeff.round(Round)
            ridge_koeff.insert(0,'Koeffizienten',ridge_koeff.index.astype(str))
            ridge_Pred.replace(0, 0.0, inplace=True)
            ridge_Fit.replace(0, 0.0, inplace=True)
            ridge_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.ridge} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.ridge} wurde nicht ausgewählt.', style={'color':'grey'})
            ridge_koeff = pd.DataFrame(columns=['Koeffizienten','Ridge'])
    else:
        ridge_koeff = pd.DataFrame(columns=['Koeffizienten','Ridge'])
        print(f'Drücken Sie den Knopf (Ridge clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (Ridge clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [ridge_Pred.to_dict('records'), ridge_Fit.to_dict('records')]
    } 
    para_store = {
        'data': ridge_koeff.to_dict('records')
    }
    print(f'ridge returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.ridge] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.ElasticNet}Store','data'),
    Output(f'{methodname.ElasticNet}ParaStore','data'),
    Output('net_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_ElasticNet(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    
    # _running_calculations[methodname.ElasticNet] = True

    net_Pred = pd.DataFrame(columns=['Date',methodname.ElasticNet])
    net_Fit = pd.DataFrame(columns=['Date',methodname.ElasticNet])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.ElasticNet in choice:
##############################ElasticNet-Methode    
            net_pred, net_fit, net_koeff = ElasticNet(y_train, exo_train, exo_test)
            net_Pred[methodname.ElasticNet] = net_pred.round(Round)
            net_Pred['Date'] = net_pred.index
            net_Pred['Date'] = net_Pred['Date'].dt.strftime(Anzeige)
            net_Pred = net_Pred[['Date',methodname.ElasticNet]]
            net_Fit[methodname.ElasticNet] = net_fit.round(Round)
            net_Fit['Date'] = net_fit.index
            net_Fit['Date'] = net_Fit['Date'].dt.strftime(Anzeige)
            net_Fit = net_Fit[['Date',methodname.ElasticNet]]
            net_koeff = net_koeff.round(Round)
            net_koeff.insert(0,'Koeffizienten',net_koeff.index.astype(str))
            net_Pred.replace(0, 0.0, inplace=True)
            net_Fit.replace(0, 0.0, inplace=True)
            net_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.ElasticNet} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.ElasticNet} wurde nicht ausgewählt.', style={'color':'grey'})
            net_koeff = pd.DataFrame(columns=['Koeffizienten','ElasticNet'])
    else:
        net_koeff = pd.DataFrame(columns=['Koeffizienten','ElasticNet'])
        print(f'Drücken Sie den Knopf (ElasticNet clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (ElasticNet clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [net_Pred.to_dict('records'), net_Fit.to_dict('records')]
    } 
    para_store = {
        'data': net_koeff.to_dict('records')
    }
    print(f'elasticNet returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.ElasticNet] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.Sarima}Store','data'),
    Output(f'{methodname.Sarima}ParaStore','data'),
    Output('sarima_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('forecastingLabelWidthStore','data'),
    State('unitsStore', 'data'),
    State('granularityStore', 'data'),
    State('seasonalityStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_Sarima(clicks, choice, ytrain, lof, unit, gran, season, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
        
    # _running_calculations[methodname.Sarima] = True

    Sarima_Pred = pd.DataFrame(columns=['Date',methodname.Sarima])
    Sarima_Fit = pd.DataFrame(columns=['Date',methodname.Sarima])
    para_df = pd.DataFrame()
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        lenght_of_forecast = lof.get('data')
        if methodname.Sarima in choice:
            # Parameters for auto_arima, more can go here
            Trend = 'Alle' # Deterministischer Trend. Optionen: None, 'c', 't', 'ct', 'Alle'
            Methode = 'lbfgs' # interner Optimierer
            opt = False
            max_order = 10
            seasonality = season.get('data')
            granularity = gran.get('data')
            freq = 'D' if granularity == 'Tage' else 'MS'
            Sarima_pred, Sarima_fit, chosen_trend, Treiber, result = AutoSARIMAX(
                y_train, Steps=int(lenght_of_forecast), 
                Trend=Trend, Methode=Methode, Optimierung=opt, max_order=max_order, 
                Freq=freq, Season=seasonality
            )
            Sarima_Pred[methodname.Sarima] = Sarima_pred.round(Round)
            Sarima_Pred['Date'] = Sarima_pred.index
            Sarima_Pred['Date'] = Sarima_Pred['Date'].dt.strftime(Anzeige)
            Sarima_Fit[methodname.Sarima] = Sarima_fit.round(Round)
            Sarima_Fit['Date'] = Sarima_fit.index
            Sarima_Fit['Date'] = Sarima_Fit['Date'].dt.strftime(Anzeige)
            para_df[methodname.Sarima] = result.params().round(4)
            para_df['Standardfehler'] = result.bse().round(3)
            para_df['p-Wert'] = result.pvalues().round(3)
            para_df.loc['Order',methodname.Sarima] = str(result.order)
            para_df.loc['Seasonal Order',methodname.Sarima] = str(result.seasonal_order)
            para_df.loc['Trend',methodname.Sarima] = chosen_trend
            para_df.loc['AIC',methodname.Sarima] = result.aic().round(3)
            para_df.loc['BIC',methodname.Sarima] = result.bic().round(3)
            para_df['Parameter'] = para_df.index
            para_df = para_df[['Parameter']+para_df.columns[:-1].to_list()]
            Sarima_Pred.replace(0, 0.0, inplace=True)
            Sarima_Fit.replace(0, 0.0, inplace=True)
            para_df.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.Sarima} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.Sarima} wurde nicht ausgewählt.', style={'color':'grey'})
    else:
        print(f'Drücken Sie den Knopf (Sarima clicks = {clicks})')
        msg = dbc.Label('Drücken Sie den Knopf (Sarima)', style={'color':'grey'})
    store = {
        'data': [Sarima_Pred.to_dict('records'), Sarima_Fit.to_dict('records')]
    }
    para_store = {
        'data': para_df.to_dict('records')
    }
    # _running_calculations[methodname.Sarima] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.Sarimax}Store','data'),
    Output(f'{methodname.Sarimax}ParaStore','data'),
    Output('sarimax_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('forecastingLabelWidthStore','data'),
    State('unitsStore', 'data'),
    State('granularityStore', 'data'),
    State('seasonalityStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_Sarimax(clicks, choice, ytrain, exotrain, exotest, lof, unit, gran, season, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
        
    # _running_calculations[methodname.Sarimax] = True

    Sarimax_Pred = pd.DataFrame(columns=['Date',methodname.Sarimax])
    Sarimax_Fit = pd.DataFrame(columns=['Date',methodname.Sarimax])
    para_df = pd.DataFrame()
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        lenght_of_forecast = lof.get('data')
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.Sarimax in choice:
            # Parameters for auto_arima, more can go here
            Trend = 'Alle' # Deterministischer Trend. Optionen: None, 'c', 't', 'ct', 'Alle'
            Methode = 'lbfgs' # interner Optimierer
            opt = False
            max_order = 10
            seasonality = season.get('data')
            granularity = gran.get('data')
            freq = 'D' if granularity == 'Tage' else 'MS'
            Sarimax_pred, Sarimax_fit, chosen_trend, Treiber, result = AutoSARIMAX(
                y_train, exo_train, exo_test, Steps=int(lenght_of_forecast),
                Trend=Trend, Methode=Methode, Optimierung=opt, max_order=max_order,
                Freq=freq, Season=seasonality
            )
            Sarimax_Pred[methodname.Sarimax] = Sarimax_pred.round(Round)
            Sarimax_Pred['Date'] = Sarimax_pred.index
            Sarimax_Pred['Date'] = Sarimax_Pred['Date'].dt.strftime(Anzeige)
            Sarimax_Fit[methodname.Sarimax] = Sarimax_fit.round(Round)
            Sarimax_Fit['Date'] = Sarimax_fit.index
            Sarimax_Fit['Date'] = Sarimax_Fit['Date'].dt.strftime(Anzeige)
            para_df[methodname.Sarimax] = result.params().round(4)
            para_df['Standardfehler'] = result.bse().round(3)
            para_df['p-Werte'] = result.pvalues().round(3)
            para_df.loc['Order',methodname.Sarimax] = str(result.order)
            para_df.loc['Seasonal Order',methodname.Sarimax] = str(result.seasonal_order)
            para_df.loc['Trend',methodname.Sarimax] = chosen_trend
            para_df.loc['AIC',methodname.Sarimax] = result.aic().round(3)
            para_df.loc['BIC',methodname.Sarimax] = result.bic().round(3)
            para_df['Parameter'] = para_df.index
            para_df = para_df[['Parameter']+para_df.columns[:-1].to_list()]
            Sarimax_Pred.replace(0, 0.0, inplace=True)
            Sarimax_Fit.replace(0, 0.0, inplace=True)
            para_df.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.Sarimax} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.Sarimax} wurde nicht ausgewählt.', style={'color':'grey'})
    else:
        print(f'Drücken Sie den Knopf (Sarimax clicks = {clicks})')
        msg = dbc.Label('Drücken Sie den Knopf (Sarimax)', style={'color':'grey'})
    store = {
        'data': [Sarimax_Pred.to_dict('records'), Sarimax_Fit.to_dict('records')]
    }
    para_store = {
        'data': para_df.to_dict('records')
    }
    # _running_calculations[methodname.Sarimax] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.XGB_RF}Store','data'),
    Output(f'{methodname.XGB_RF}ParaStore','data'),
    Output('xgbrf_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_XGB_RF(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    # _running_calculations[methodname.XGB_RF] = True

    XGB_RF_Pred = pd.DataFrame(columns=['Date',methodname.XGB_RF])
    XGB_RF_Fit = pd.DataFrame(columns=['Date',methodname.XGB_RF])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.XGB_RF in choice:
##############################XGB_RF-Methode    
            XGB_RF_pred, XGB_RF_fit, XGB_RF_koeff, XGB_RF_Para = XGB_RandomForest(y_train, exo_train, exo_test, 1)
            XGB_RF_Pred[methodname.XGB_RF] = XGB_RF_pred[methodname.XGB_RF].round(Round)
            XGB_RF_Pred['Date'] = XGB_RF_pred.index
            XGB_RF_Pred['Date'] = XGB_RF_Pred['Date'].dt.strftime(Anzeige)
            XGB_RF_Fit[methodname.XGB_RF] = XGB_RF_fit[methodname.XGB_RF].round(Round)
            XGB_RF_Fit['Date'] = XGB_RF_fit.index
            XGB_RF_Fit['Date'] = XGB_RF_Fit['Date'].dt.strftime(Anzeige)
            XGB_RF_koeff = XGB_RF_koeff.round(6)
            XGB_RF_koeff.insert(0,'exogene Variablen',XGB_RF_koeff.index.astype(str))
            XGB_RF_Pred.replace(0, 0.0, inplace=True)
            XGB_RF_Fit.replace(0, 0.0, inplace=True)
            XGB_RF_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.XGB_RF} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.XGB_RF} wurde nicht ausgewählt.', style={'color':'grey'})
            XGB_RF_koeff = pd.DataFrame(columns=['exogene Variablen',methodname.XGB_RF])
    else:
        XGB_RF_koeff = pd.DataFrame(columns=['exogene Variablen',methodname.XGB_RF])
        print(f'Drücken Sie den Knopf (XGB_RF clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (XGB_RF clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [XGB_RF_Pred.to_dict('records'), XGB_RF_Fit.to_dict('records')]
    } 
    para_store = {
        'data': XGB_RF_koeff.to_dict('records')
    }
    print(f'XGB_RF returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.XGB_RF] = False
    return store, para_store, msg

@callback(
    Output(f'{methodname.GB}Store','data'),
    Output(f'{methodname.GB}ParaStore','data'),
    Output('gb_message', 'children'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('variableTrainStore', 'data'),
    State('exoTrainStore','data'),
    State('exoTestStore','data'),
    State('unitsStore', 'data'),
    Input('initialText','children'),
    # background=True,
    # manager=background_callback_manager,
)
def method_GB(clicks, choice, ytrain, exotrain, exotest, unit, childrenIni):
    if choice == 0 or ytrain == {} or clicks == 0:
        raise PreventUpdate
    if childrenIni != 'Methoden starten':
        raise PreventUpdate
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = 'in €'
        Round = 2
    elif units == 'Stückzahl Stk':
        Angabe = 'in Stk'
        Round = 0
    elif units == 'einheitslos':
        Angabe = ''
        Round = 9
    Anzeige = '%d-%m-%Y'
    # _running_calculations[methodname.GB] = True

    GB_Pred = pd.DataFrame(columns=['Date',methodname.GB])
    GB_Fit = pd.DataFrame(columns=['Date',methodname.GB])
    if clicks > 0:
        y_train = pd.DataFrame.from_dict(ytrain.get('data'))
        y_train.set_index('Date', inplace=True)
        y_train.index = pd.to_datetime(y_train.index, format=Anzeige)
        
        exo_train = pd.DataFrame.from_dict(exotrain.get('data'))
        exo_train.set_index('Date', inplace=True)
        exo_train.index = pd.to_datetime(exo_train.index, format=Anzeige)
        
        exo_test = pd.DataFrame.from_dict(exotest.get('data'))
        exo_test.set_index('Date', inplace=True)
        exo_test.index = pd.to_datetime(exo_test.index, format=Anzeige)
        if methodname.GB in choice:
##############################GB-Methode    
            GB_pred, GB_fit, GB_koeff, GB_Para = XGBoost(y_train, exo_train, exo_test,1)
            GB_Pred[methodname.GB] = GB_pred[methodname.GB].round(Round)
            GB_Pred['Date'] = GB_pred.index
            GB_Pred['Date'] = GB_Pred['Date'].dt.strftime(Anzeige)
            GB_Fit[methodname.GB] = GB_fit[methodname.GB].round(Round)
            GB_Fit['Date'] = GB_fit.index
            GB_Fit['Date'] = GB_Fit['Date'].dt.strftime(Anzeige)
            GB_koeff = GB_koeff.round(6)
            GB_koeff.insert(0,'exogene Variablen',GB_koeff.index.astype(str))
            GB_Pred.replace(0, 0.0, inplace=True)
            GB_Fit.replace(0, 0.0, inplace=True)
            GB_koeff.replace(0, 0.0, inplace=True)
            msg = dbc.Label(f'Methode {methodname.GB} ist fertig', style={'color':'green'})
        else:
            msg = dbc.Label(f'Methode {methodname.GB} wurde nicht ausgewählt.', style={'color':'grey'})
            GB_koeff = pd.DataFrame(columns=['exogene Variablen',methodname.GB])
    else:
        GB_koeff = pd.DataFrame(columns=['exogene Variablen',methodname.GB])
        print(f'Drücken Sie den Knopf (GB clicks = {clicks})')
        msg = dbc.Label(f'Drücken Sie den Knopf (GB clicks = {clicks})', style={'color':'grey'})
    store = {
        'data': [GB_Pred.to_dict('records'), GB_Fit.to_dict('records')]
    } 
    para_store = {
        'data': GB_koeff.to_dict('records')
    }
    print(f'GB returning {time.strftime("%d.%m.%Y %H:%M:%S")}')
    # _running_calculations[methodname.GB] = False
    return store, para_store, msg

@callback(
    Output('predictionsStore','data'),
    Output('prediction-button', 'disabled'),
    Output('results-button', 'disabled'),
    Output("Uploader", "disabled"),
    Output("Voreinstellungen", "disabled"),
    Output("Methodenauswahl", "disabled"), 
    Output("Produktauswahl", "disabled"), 
    Output("Treiberauswahl", "disabled"),
    Output("Zusammenfassung", "disabled"),
    Output('cancel-button', 'disabled', allow_duplicate=True),
    Output('cancel-button', 'color', allow_duplicate=True),
    Output('cancel_button_reminder', 'children', allow_duplicate=True),
    Output('initialText','children', allow_duplicate=True),
    Output('display-selection', 'options', allow_duplicate=True),
    Output('display-selection', 'value', allow_duplicate=True),
    Output('results-button', 'n_clicks'),
    State(f'{methodname.Holt_Winters}Store','data'),
    State(f'{methodname.LASSO}Store','data'),
    State(f'{methodname.mlr}Store','data'),
    State(f'{methodname.ridge}Store','data'),
    State(f'{methodname.ElasticNet}Store','data'),
    State(f'{methodname.Sarima}Store','data'),
    State(f'{methodname.Sarimax}Store','data'),
    State(f'{methodname.XGB_RF}Store','data'),
    State(f'{methodname.GB}Store','data'),
    State('prediction-button','n_clicks'),
    State('method-selection', 'value'),
    State('initialText','children'),
    Input('hw_message', 'children'),
    Input('lasso_message', 'children'),
    Input('mlr_message', 'children'),
    Input('ridge_message', 'children'),
    Input('net_message', 'children'),
    Input('sarima_message', 'children'),
    Input('sarimax_message', 'children'),
    Input('xgbrf_message', 'children'),
    Input('gb_message', 'children'),
    prevent_initial_call=True
)
def pred_Store(HoltWintersd, Lassod, Mlrd, Ridged, Netd, Sarimad, Sarimaxd, XGB_RFd, GBd, 
                clicks, choice, childrenIni, HW_C, L_C, M_C, R_C, EN_C, S_C, SX_C, 
                XGBRF_C, GB_C):
    # if True in _running_calculations.values(): 
    #     raise PreventUpdate
    if (
        (HW_C == [] or L_C == [] or M_C == [] or R_C == [] or EN_C == [] or S_C == [] or SX_C == []
        or XGBRF_C == [] or GB_C == []) or
        (HW_C['props']['style']['color'] == 'tomato' or L_C['props']['style']['color'] == 'tomato' or 
        M_C['props']['style']['color'] == 'tomato' or R_C['props']['style']['color'] == 'tomato' or 
        EN_C['props']['style']['color'] == 'tomato' or S_C['props']['style']['color'] == 'tomato' or 
        SX_C['props']['style']['color'] == 'tomato' or XGBRF_C['props']['style']['color'] == 'tomato' or
        GB_C['props']['style']['color'] == 'tomato')
        ):
        raise PreventUpdate      
    if choice is None:
        print(f'Wählen Sie die gewünschte/n Methode/n aus. pred_Store clicks = {clicks}')
        raise PreventUpdate
    store_list = {
        methodname.Holt_Winters: HoltWintersd, methodname.LASSO: Lassod, methodname.mlr: Mlrd, 
        methodname.ridge: Ridged, methodname.ElasticNet: Netd, methodname.Sarima: Sarimad,
        methodname.Sarimax: Sarimaxd, methodname.XGB_RF: XGB_RFd, methodname.GB: GBd
    }
    pred_method = pd.DataFrame(columns=['Date']+choice)
    fit_method = pd.DataFrame(columns=['Date']+choice)
    for method in choice:
        m_store = store_list[method]
        pred_df = pd.DataFrame.from_dict(m_store.get('data')[0])
        fit_df = pd.DataFrame.from_dict(m_store.get('data')[1])
        pred_method[method] = pred_df[method]
        pred_method['Date'] = pred_df['Date']
        fit_method[method] = fit_df[method]
        fit_method['Date'] = fit_df['Date']
            
    store = {
        'data': [pred_method.to_dict('records'), fit_method.to_dict('records')]
    }
    d = False
    childrenIni = 'Methoden sind durchgelaufen'
    d_opt = [
        {'value':'pred','label':'Prognose','disabled':False},
        {'value':'metr','label':'Metriken','disabled':False},
        {'value':'fit','label':'Trainingsgüte','disabled':False},
        {'value':'para','label':'Parameterwerte/Treiberbewertung','disabled':False},
    ]
    return store, d, d, d, d, d, d, d, d, not d, 'secondary', '', childrenIni, d_opt, 'pred', 1

@callback(
    Output('predContainer', 'hidden'),
    Output('metrikContainer', 'hidden'),
    Output('fitContainer', 'hidden'),
    Output('paramContainer', 'hidden'),
    Output(methodname.Holt_Winters, 'hidden'),
    Output(methodname.LASSO, 'hidden'),
    Output(methodname.mlr, 'hidden'),
    Output(methodname.ridge, 'hidden'),
    Output(methodname.ElasticNet, 'hidden'),
    Output(methodname.Sarima, 'hidden'),
    Output(methodname.Sarimax, 'hidden'),
    Output(methodname.XGB_RF, 'hidden'),
    # Output(methodname.RF, 'hidden'),
    Output(methodname.GB, 'hidden'),
    # Output(methodname.LGB, 'hidden'),
    Output('MetrikMAPE', 'hidden'),
    Output('MetrikMAE', 'hidden'),
    Output('MetrikSMAPE', 'hidden'),
    Output('MetrikMFPE', 'hidden'),
    Output('MetrikMFE', 'hidden'),
    Output('hw_message', 'children', allow_duplicate=True),
    Output('lasso_message', 'children', allow_duplicate=True),
    Output('mlr_message', 'children', allow_duplicate=True),
    Output('ridge_message', 'children', allow_duplicate=True),
    Output('net_message', 'children', allow_duplicate=True),
    Output('sarima_message', 'children', allow_duplicate=True),
    Output('sarimax_message', 'children', allow_duplicate=True),
    Output('xgbrf_message', 'children', allow_duplicate=True),
    # Output('rf_message', 'children', allow_duplicate=True),
    Output('gb_message', 'children', allow_duplicate=True),
    # Output('lgb_message', 'children', allow_duplicate=True),
    Input('display-selection', 'value'),
    Input('param-display-selection', 'value'),
    Input('metric-display-selection', 'value'),
    Input('results-button', 'n_clicks'),
    prevent_initial_call = True
)
def toggle_dispay(choice, para_choice, metric_choice, clicks):
    if clicks==0 or choice is None or para_choice is None:
        raise PreventUpdate
    return ('pred' != choice, 'metr' != choice, 'fit' != choice, 'para' != choice,
            methodname.Holt_Winters != para_choice, methodname.LASSO != para_choice, 
            methodname.mlr != para_choice, methodname.ridge != para_choice, 
            methodname.ElasticNet != para_choice, methodname.Sarima != para_choice, 
            methodname.Sarimax != para_choice, methodname.XGB_RF != para_choice, 
            # methodname.RF != para_choice, methodname.GB != para_choice, 
            # methodname.LGB != para_choice, 
            methodname.GB != para_choice, # 
            'MAPE' != metric_choice, 'MAE' != metric_choice, 'SMAPE' != metric_choice,
            'MFPE' != metric_choice, 'MFE' != metric_choice, 
            # [], [], [], [], [], [], [], [], [], [], [])
            [], [], [], [], [], [], [], [], [])

#Grafik erstellen mit ausgewählten Methoden    
@callback(
    Output('Grafik','children'),
    Output('Fit_Grafik','children'),
    Output('PredictionTable','data'),
    Output('PredictionTable','page_size'),
    Output('PredTabWidth','width'),
    State('prediction-button','n_clicks'),
    State('method-selection','value'),
    State('productStore','data'),
    State('productChoice','data'),
    State('trainingDateStore', 'data'),
    State('forecastDateStore', 'data'),
    State('forecastingLabelWidthStore','data'),
    State('unitsStore', 'data'),
    State('granularityStore', 'data'),
    Input('results-button','n_clicks'),
    State('predictionsStore','data'),
    State('SzenarioStore', 'data'),
    State('productChoice','modified_timestamp'),
    State('driverChoice', 'modified_timestamp'),
    State('predictionsStore', 'modified_timestamp'),
)
def show_table_and_graph(clicks, choice, productStore, productc, Datemin, Datepred, pred_len, unit, gran,
                        rClicks, predStore,szenarioStore, prod_time, drive_time, pred_time): 
    units = unit.get('data')
    if units == 'Euro €':
        Angabe = ' in €'
    elif units == 'Stückzahl Stk':
        Angabe = ' in Stk'
    elif units == 'einheitslos':
        Angabe = ''
    Anzeige = '%d-%m-%Y'
        
    if ((clicks == 0 and (szenarioStore is None or szenarioStore == {})) 
        or prod_time is None or drive_time is None or pred_time is None
        or (prod_time > pred_time) or (drive_time > pred_time)):
        raise PreventUpdate

    product_df = pd.DataFrame.from_dict(productStore.get('data'))
    product_df.set_index('Date', inplace=True)
    product_df.index = pd.to_datetime(product_df.index, format=Anzeige)
    product_df = product_df.astype(float)
    product_choice = productc.get('data')
    Date_min = Datemin.get('data')
    Date_pred = Datepred.get('data')
    pro_train = pd.DataFrame(columns=[product_choice])
    pro_train[product_choice] = product_df[product_choice][(product_df.index >= pd.to_datetime(Date_min, format=Anzeige))&
                                                     (product_df.index < pd.to_datetime(Date_pred, format=Anzeige))]
    temp = pd.DataFrame(columns=[product_choice])
    temp[product_choice] = product_df[product_choice][(product_df.index >= pd.to_datetime(Date_pred, format=Anzeige))]
    if len(temp)>=pred_len['data']:
        pro = pd.concat([pro_train, temp.loc[temp.index[0:pred_len['data']]]])
    else:
        pro = pd.concat([pro_train,temp])
    pro['Date'] = pro.index
    pro_train['Date'] = pro_train.index

    fig = make_subplots(
        specs= [[{'type':'scatter'}]]
        )
    fig.add_trace(
        go.Scatter(
            x=pro['Date'],
            y=pro[product_choice],
            mode='markers+lines',
            name= product_choice),
        )
    if not (choice is None or predStore is None or predStore == {} or rClicks==0): 
        pred_method = pd.DataFrame.from_dict(predStore.get('data')[0])
        pred_method.set_index('Date', inplace=True)
        pred_method.index = pd.to_datetime(pred_method.index, format=Anzeige)
        pred_method['Date'] = pred_method.index
        for method in choice:
            if method in pred_method.columns.to_list():
                fig.add_trace(
                    go.Scatter(
                        x=pred_method['Date'],
                        y=pred_method[method],
                        mode='markers+lines',
                        name= method),
                    )
    if not szenarioStore is None and szenarioStore != {} and szenarioStore.get('data') != []:
        szenario = pd.DataFrame.from_dict(szenarioStore.get('data'))
        szenario.set_index('Date', inplace=True)
        szenario.index = pd.to_datetime(szenario.index, format=Anzeige)
        szenario['Date'] = szenario.index
        col = szenario.columns.to_list()
        col.remove('Date')
        for i in col:
            fig.add_trace(
                go.Scatter(
                        x=szenario['Date'],
                        y=szenario[i],
                        mode='markers+lines',
                        name= i)
            )
    fig.update_layout(
        showlegend = True,
        title_text=f'Prognosemethoden für {product_choice} {Angabe}',
        height = 450,
        yaxis_tickformat = ',~r'
        )
    
    fit_fig = make_subplots(
        specs= [[{'type':'scatter'}]]
        )
    fit_fig.add_trace(
        go.Scatter(
            x=pro_train['Date'],
            y=pro_train[product_choice],
            mode='markers+lines',
            name= product_choice),
        )
    if not (choice is None or predStore is None or predStore == {} or rClicks==0): 
        fit_method = pd.DataFrame.from_dict(predStore.get('data')[1])
        fit_method.set_index('Date', inplace=True)
        fit_method.index = pd.to_datetime(fit_method.index, format=Anzeige)
        fit_method['Date']=pd.to_datetime(fit_method.index)
        for method in choice:
            if method in fit_method.columns.to_list():
                fit_fig.add_trace(
                    go.Scatter(
                        x=fit_method['Date'],
                        y=fit_method[method],
                        mode='markers+lines',
                        name= method),
                    )
    fit_fig.update_layout(
        showlegend = True,
        title_text=f'Reproduktion Trainingsdaten durch Prognosemethoden',
        height = 450,
        yaxis_tickformat = ',~r'
        )
    if rClicks > 0:
        children = dcc.Graph(id='GrafikTabelleMethode', figure=fig) 
        fit_children = dcc.Graph(id='FitGrafikTabelleMethode', figure=fit_fig) 
        style = 2
        if not choice is None: 
            style += len(choice)
        if not szenarioStore is None and szenarioStore != {} and szenarioStore.get('data') != []:
            style += len(szenarioStore['data'][0].keys())-1
        if not predStore is None and predStore != {}: 
            table = pd.DataFrame.from_dict(predStore.get('data')[0])
            if szenarioStore != {} and szenarioStore.get('data') != []:
                table_szenario = pd.DataFrame.from_dict(szenarioStore.get('data'))
                table = table.merge(table_szenario, how='inner', on='Date')
        else: 
            table = pd.DataFrame.from_dict(szenarioStore.get('data'))
        granularity = gran.get('data')
        gran_num = 7 if granularity=='Tage' else 12 # ggf noch anpassen für andere Saisonalität
        return children, fit_children, table.to_dict('records'), min(pred_len.get('data'), gran_num), style
    else:
        return [], [], None, no_update, no_update

@callback(
    Output('MetrikMAPE','children'),
    Output('MetrikMAE','children'),
    Output('MetrikSMAPE','children'),
    Output('MetrikMFPE','children'),
    Output('MetrikMFE','children'),
    Output('Fit_Metrik','children'),
    State('prediction-button','n_clicks'),
    State('predictionsStore','data'),
    State('SzenarioStore','data'),
    State('productStore','data'),
    State('productChoice','data'),
    State('forecastDateStore', 'data'),
    State('trainingDateStore', 'data'),
    State('forecastingLabelWidthStore','data'),
    State('granularityStore', 'data'),
    Input('results-button', 'n_clicks'),
    State('productChoice','modified_timestamp'),
    State('driverChoice', 'modified_timestamp'),
    State('predictionsStore', 'modified_timestamp'),
    State('exoTestStore', 'data'),
    # background=True,
    # manager=background_callback_manager,
)

def show_MAPE(clicks, pred_m, szenarioStore, productStore, productc, Datec, DateT, lof, gran, 
              rClicks, prod_time, drive_time, pred_time, exoTest):
    if (rClicks == 0 or prod_time is None or drive_time is None or pred_time is None
        or (prod_time > pred_time) or (drive_time > pred_time)):
        raise PreventUpdate
    else:
        Anzeige = "%d-%m-%Y"
        pred_method = pd.DataFrame.from_dict(pred_m.get('data')[0])
        fit_method = pd.DataFrame.from_dict(pred_m.get('data')[1])
        szenario = pd.DataFrame.from_dict(szenarioStore.get('data'))
        gran = gran.get('data')
        tab_width = 2            
        if not pred_m is None and pred_m != {}:
            pred_method.set_index('Date', inplace=True, drop=True)
            pred_method = pred_method.astype(float)
            fit_method.set_index('Date', inplace=True, drop=True)
            fit_method = fit_method.astype(float)
        if not szenarioStore is None and szenarioStore != {}:
            szenario.set_index('Date', inplace=True, drop=True)
            szenario = szenario.astype(float)
            szenario_lst = szenario.columns.to_list()
            tab_width += len(szenario_lst)
        product_df = pd.DataFrame.from_dict(productStore.get('data'))
        product_df.set_index('Date', inplace=True)
        product_df.index = pd.to_datetime(product_df.index, format=Anzeige)
        product_df = product_df.astype(float)
        product_choice = productc.get('data')
        Date_cut = Datec.get('data')
        Date_train = DateT.get('data')
        lenght_of_forecast = lof.get('data')
        y_true = pd.DataFrame()
        y_true[product_choice] = product_df[product_choice][(product_df.index >= pd.to_datetime(Date_cut, format=Anzeige))]
        y_true = y_true[product_choice][0:lenght_of_forecast]
        if y_true.shape[0] > 0:
            y_true.index = pd.Index(y_true.index.to_series().dt.strftime(Anzeige))
        if y_true.shape[0] < lenght_of_forecast:
            y_true = pd.Series(y_true, index=pd.DataFrame.from_dict(exoTest['data'])['Date'])
        y_train = pd.DataFrame()
        y_train[product_choice] = product_df[product_choice][((product_df.index >= pd.to_datetime(Date_train, format=Anzeige))&
                                                              (product_df.index < pd.to_datetime(Date_cut, format=Anzeige)))]
        y_train = y_train[product_choice][:]
        metric_results_mape = {}
        metric_results_mape_pm_df = pd.DataFrame(index=y_true.index)
        metric_results_mae = {}
        metric_results_mae_pm_df = pd.DataFrame(index=y_true.index)
        metric_results_smape = {}
        metric_results_smape_pm_df = pd.DataFrame(index=y_true.index)
        metric_results_mfpe = {}
        metric_results_mfpe_pm_df = pd.DataFrame(index=y_true.index)
        metric_results_mfe = {}
        metric_results_mfe_pm_df = pd.DataFrame(index=y_true.index)
        if not pred_m is None and pred_m != {}:
            lst = pred_method.columns.to_list()
            tab_width += len(lst)
            if 'Date' in lst:
                lst.remove('Date')
            for i in lst:
                mape_pm, mape_min, mape_max = metric_MAPE_separate(y_true,pred_method[i])
                metric_results_mape_pm_df[i] = mape_pm
                metric_results_mape[i]=[
                    metric_MAPE(y_true,pred_method[i]),
                    metric_MAPE_pj(y_true,pred_method[i]),
                    mape_min, 
                    mape_max
                ]
                mae_pm, mae_min, mae_max = metric_MAE_separate(y_true,pred_method[i])
                metric_results_mae_pm_df[i] = mae_pm
                metric_results_mae[i]=[
                    metric_MAE(y_true,pred_method[i]),
                    metric_MAE_pj(y_true,pred_method[i]),
                    mae_min, 
                    mae_max
                ]
                smape_pm, smape_min, smape_max = metric_SMAPE_separate(y_true,pred_method[i])
                metric_results_smape_pm_df[i] = smape_pm
                metric_results_smape[i]=[
                    metric_SMAPE(y_true,pred_method[i]),
                    metric_SMAPE_pj(y_true,pred_method[i]),
                    smape_min, 
                    smape_max
                ]
                mfpe_pm = metric_rel_bias_separate(y_true,pred_method[i])
                metric_results_mfpe_pm_df[i] = mfpe_pm
                metric_results_mfpe[i]=[
                    metric_rel_MFE(y_true,pred_method[i]),
                    metric_rel_total_bias(y_true,pred_method[i])
                ]
                mfe_pm = metric_bias_separate(y_true,pred_method[i])
                metric_results_mfe_pm_df[i] = mfe_pm
                metric_results_mfe[i]=[
                    metric_MFE(y_true,pred_method[i]),
                    metric_total_bias(y_true,pred_method[i]),
                ]
        if not szenarioStore is None and szenarioStore != {}:
            for j in szenario_lst:
                mape_pm, mape_min, mape_max = metric_MAPE_separate(y_true,szenario[j])
                metric_results_mape_pm_df[j] = mape_pm
                metric_results_mape[j]=[
                    metric_MAPE(y_true,szenario[j]),
                    metric_MAPE_pj(y_true,szenario[j]), 
                    mape_min, 
                    mape_max
                ]
                mae_pm, mae_min, mae_max = metric_MAE_separate(y_true,szenario[j])
                metric_results_mae_pm_df[j] = mae_pm
                metric_results_mae[j]=[
                    metric_MAE(y_true,szenario[j]),
                    metric_MAE_pj(y_true,szenario[j]), 
                    mae_min, 
                    mae_max
                ]
                smape_pm, smape_min, smape_max = metric_SMAPE_separate(y_true,szenario[j])
                metric_results_smape_pm_df[j] = smape_pm
                metric_results_smape[j]=[
                    metric_SMAPE(y_true,szenario[j]),
                    metric_SMAPE_pj(y_true,szenario[j]), 
                    smape_min, 
                    smape_max
                ]
                mfpe_pm = metric_rel_bias_separate(y_true,szenario[j])
                metric_results_mfpe_pm_df[j] = mfpe_pm
                metric_results_mfpe[j]=[
                    metric_rel_MFE(y_true,szenario[j]),
                    metric_rel_total_bias(y_true,szenario[j])
                ]
                mfe_pm = metric_bias_separate(y_true,szenario[j])
                metric_results_mfe_pm_df[j] = mfe_pm
                metric_results_mfe[j]=[
                    metric_MFE(y_true,szenario[j]),
                    metric_total_bias(y_true,szenario[j]),
                ]
        if metric_results_mape == {}:
            raise PreventUpdate
        metric_results_mape_df = pd.DataFrame.from_dict(
            metric_results_mape, orient='index', 
            columns=[f"MAPE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", "MAPE Gesamtprognose", "MAPE Min", "MAPE Max"], 
        )
        metric_results_mape_df = metric_results_mape_df.reset_index(names='Methoden/Szenarien')
        metric_results_mape_pm_df.insert(0, 'Datum', metric_results_mape_pm_df.index)
        metric_results_smape_df = pd.DataFrame.from_dict(
            metric_results_smape, orient='index', 
            columns=[f"SMAPE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", "SMAPE Gesamtprognose", "SMAPE Min", "SMAPE Max"], 
        )
        metric_results_smape_df = metric_results_smape_df.reset_index(names='Methoden/Szenarien')
        metric_results_smape_pm_df.insert(0, 'Datum', metric_results_smape_pm_df.index)
        metric_results_mae_df = pd.DataFrame.from_dict(
            metric_results_mae, orient='index', 
            columns=[f"MAE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", "MAE Gesamtprognose", "MAE Min", "MAE Max"], 
        )
        metric_results_mae_df = metric_results_mae_df.reset_index(names='Methoden/Szenarien')
        metric_results_mae_pm_df.insert(0, 'Datum', metric_results_mae_pm_df.index)
        metric_results_mfpe_df = pd.DataFrame.from_dict(
            metric_results_mfpe, orient='index', 
            columns=[f"MFPE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", "Relativer Bias der Gesamtprognose"], 
        )
        metric_results_mfpe_df = metric_results_mfpe_df.reset_index(names='Methoden/Szenarien')
        metric_results_mfpe_pm_df.insert(0, 'Datum', metric_results_mfpe_pm_df.index)
        metric_results_mfe_df = pd.DataFrame.from_dict(
            metric_results_mfe, orient='index', 
            columns=[f"MFE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", "Bias der Gesamtprognose"], 
        )
        metric_results_mfe_df = metric_results_mfe_df.reset_index(names='Methoden/Szenarien')
        metric_results_mfe_pm_df.insert(0, 'Datum', metric_results_mfe_pm_df.index)
        fit_metric_results = {}
        if not pred_m is None and pred_m != {}:
            lst = fit_method.columns.to_list()
            if 'Date' in lst:
                lst.remove('Date')
            for i in lst:
                fit_metric_results[i]=[
                    metric_MAPE(y_train,fit_method[i]),
                    metric_MAPE_pj(y_train,fit_method[i]),
                    metric_MAE(y_train,fit_method[i]),
                    metric_MAE_pj(y_train,fit_method[i]),
                    metric_SMAPE(y_train,fit_method[i]),
                    metric_SMAPE_pj(y_train,fit_method[i]),
                ]
        if fit_metric_results == {}:
            raise PreventUpdate
        fit_metric_results_df = pd.DataFrame.from_dict(
            fit_metric_results, orient='index', 
            columns=[f"MAPE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", 'MAPE Gesamtprognose',
                     f"MAE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", 'MAE Gesamtprognose',
                     f"SMAPE Durchschnitt/{'Tag' if gran=='Tage' else 'Monat'}", 'SMAPE Gesamtprognose',], 
        )
        fit_metric_results_df = fit_metric_results_df.reset_index(names='Methoden')
        MAPE_children = [dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die den Prozentualen Fehler (MAPE*) für die verschiedenen Methoden vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [    
                            dash_table.DataTable(metric_results_mape_df.to_dict('records'), 
                                                    id = 'Metric_Tabelle_MAPE',
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto',},) 
                            ], width = 6
                        ),
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dbc.Label(f'Prozentualer Fehler für alle {gran} separat', color='secondary')
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [
                            dash_table.DataTable(metric_results_mape_pm_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_MAPE_pm',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                            ], width = tab_width
                        )
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row(
                        dcc.Markdown(
                            '''
                            \* Fehler der Prognose anteilig an Produktdaten  
                            Nachteil: Bestraft Überschätzen stärker als Unterschätzen  
                            Für zuverlässige Fehlerbeurteilung mit SMAPE und MAE vergleichen. 
                            ''',
                            style={'fontStyle':'italic'}
                        )
                    ),
        ]
        SMAPE_children = [dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die den Symmetrischen Prozentualen Fehler (SMAPE*) für die verschiedenen Methoden vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [    
                            dash_table.DataTable(metric_results_smape_df.to_dict('records'), 
                                                    id = 'Metric_Tabelle_SMAPE',
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto',},) 
                            ], width = 6
                        ),
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dbc.Label(f'Symmetrischer Prozentualer Fehler für alle {gran} separat', color='secondary')
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [
                            dash_table.DataTable(metric_results_smape_pm_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_SMAPE_pm',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                            ], width = tab_width
                        )
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row(
                        dcc.Markdown(
                            '''
                            \* Variation von MAPE, versucht fehlende Symmetrie zwischen  
                            Unter- und Überschätzung auszugleichen. 
                            ''',
                            style={'fontStyle':'italic'}
                        )
                    ),
        ]
        MAE_children = [dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die den Absoluten Fehler (MAE*) für die verschiedenen Methoden vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [    
                            dash_table.DataTable(metric_results_mae_df.to_dict('records'), 
                                                    id = 'Metric_Tabelle_MAE',
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto',},) 
                            ], width = 6
                        ),
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dbc.Label(f'Absoluter Fehler für alle {gran} separat', color='secondary')
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [
                            dash_table.DataTable(metric_results_mae_pm_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_MAE_pm',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                            ], width = tab_width
                        )
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row(
                        dcc.Markdown(
                            '''
                            \* Hat die selbe Einheit/Skalierung wie das Produkt. 
                            ''',
                            style={'fontStyle':'italic'}
                        )
                    ),
        ]
        MFPE_children = [dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die den Bias der Prognose/MFPE* für die verschiedenen Methoden vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [    
                            dash_table.DataTable(metric_results_mfpe_df.to_dict('records'), 
                                                    id = 'Metric_Tabelle_MFPE',
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto',},) 
                            ], width = 6
                        ),
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dbc.Label(f'Bias für alle {gran} separat', color='secondary')
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [
                            dash_table.DataTable(metric_results_mfpe_pm_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_MFPE_pm',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                            ], width = tab_width
                        )
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row(
                        dcc.Markdown(
                            '''
                            \* Prozentuale Abweichung der Prognose.  
                            Zeigt an, ob die Methoden unter- oder überschätzen.  
                            Negativer Wert - die Methode unterschätzt  
                            Positiver Wert - die Methode überschätzt
                            ''',
                            style={'fontStyle':'italic'}
                        ) 
                    ),
        ]
        MFE_children = [dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die den Bias der Prognose/MFE* für die verschiedenen Methoden vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [    
                            dash_table.DataTable(metric_results_mfe_df.to_dict('records'), 
                                                    id = 'Metric_Tabelle_MFE',
                                                    style_cell={
                                                        'textAlign': 'center',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto',},) 
                            ], width = 6
                        ),
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dbc.Label(f'Bias für alle {gran} separat', color='secondary')
                    ]),
                    dbc.Row(
                        dbc.Col(
                            [
                            dash_table.DataTable(metric_results_mfe_pm_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_MFE_pm',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                            ], width = tab_width
                        )
                    ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row(
                        dcc.Markdown(
                            '''
                            \* Hat die selbe Einheit/Skalierung wie das Produkt.  
                            Zeigt an, ob die Methoden unter- oder überschätzen.  
                            Negativer Wert - die Methode unterschätzt  
                            Positiver Wert - die Methode überschätzt 
                            ''',
                            style={'fontStyle':'italic'}
                        ) 
                    ),
        ]
        fit_children = [
                    dbc.Row(
                    [
                        dbc.Label('Vergleich der Methoden für den Trainingszeitraum: ', style={'fontSize':20, 'textAlign':'center', 'width':'300',}),
                        dbc.Label('Tabellarische Übersicht, die die Gütekriterien für die verschiedenen Methoden auf den Trainingsdaten vergleicht.', color='secondary'),
                    ]),
                    dbc.Row(
                            dbc.Col(
                        [    
                        dash_table.DataTable(fit_metric_results_df.to_dict('records'), 
                                                id = 'Metric_Tabelle_fit',
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',},) 
                        ], width =8),
                        ),
                    html.Hr(style={'visibility':'hidden'}),
                    dbc.Row([
                        dcc.Markdown(
                            '''
                            Achtung: Wenn die Fehler (insbesondere die akkumulierten) für eine der  
                            Methoden **nicht** nahe Null sind, d.h. wenn die Reproduktion weit von  
                            den Trainingsdaten entfernt ist, kann das bedeuten, dass die Prognose  
                            dieser Methode falsch ist, weil das Training nicht erfolgreich war.  
                              
                            Falls andererseits die Reproduktion "zu gut" ist und kleinste Schwankungen  
                            perfekt abbildet, kann das bedeuten, dass Phantomeffekte in die Zukunft  
                            projeziert werden.   
                            ''',
                        style={'color':'tomato'})
                    ])
        ]
        return MAPE_children, MAE_children, SMAPE_children, MFPE_children, MFE_children, fit_children
    

@callback(
    Output('metricStore', 'data'),
    Input('MetrikMAPE','children'),
    Input('MetrikMAE','children'),
    Input('MetrikSMAPE','children'),
    Input('MetrikMFPE','children'),
    Input('MetrikMFE','children')
)
def update_metricStore(mape_children, mae_children, smape_children, mfpe_children, mfe_children):
    if (not mape_children is None and not mae_children is None and not smape_children is None
        and not mfpe_children is None and not mfe_children is None):
        try:
            Tabelle_mape = [
                pd.DataFrame.from_dict(mape_children[1]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'), 
                pd.DataFrame.from_dict(mape_children[4]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'),
                ]
            Tabelle_smape = [
                pd.DataFrame.from_dict(smape_children[1]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'), 
                pd.DataFrame.from_dict(smape_children[4]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'),
                ]
            Tabelle_mae = [
                pd.DataFrame.from_dict(mae_children[1]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'), 
                pd.DataFrame.from_dict(mae_children[4]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'),
                ]
            Tabelle_mfpe = [
                pd.DataFrame.from_dict(mfpe_children[1]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'), 
                pd.DataFrame.from_dict(mfpe_children[4]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'),
                ]
            Tabelle_mfe = [
                pd.DataFrame.from_dict(mfe_children[1]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'), 
                pd.DataFrame.from_dict(mfe_children[4]['props']['children']['props']['children'][0]['props']['data']).to_dict('records'),
                ]
            Tabelle = [Tabelle_mape, Tabelle_smape, Tabelle_mae, Tabelle_mfpe, Tabelle_mfe]
        except Exception as e:
            print(e)
            Tabelle = []
    else: 
        raise PreventUpdate
    
    store = {
        'data' : Tabelle
    }
    return store

@callback(
    Output(methodname.Holt_Winters,'children', allow_duplicate=True),
    Output(methodname.LASSO,'children', allow_duplicate=True),
    Output(methodname.mlr,'children', allow_duplicate=True),
    Output(methodname.ridge,'children', allow_duplicate=True),
    Output(methodname.ElasticNet,'children', allow_duplicate=True),
    Output(methodname.Sarima, 'children', allow_duplicate=True),
    Output(methodname.Sarimax, 'children', allow_duplicate=True),
    Output(methodname.XGB_RF, 'children', allow_duplicate=True),
    Output(methodname.GB, 'children', allow_duplicate=True),
    Input('results-button', 'n_clicks'),
    State(f'{methodname.Holt_Winters}ParaStore','data'),
    State(f'{methodname.LASSO}ParaStore','data'),
    State(f'{methodname.mlr}ParaStore','data'),
    State(f'{methodname.ridge}ParaStore','data'),
    State(f'{methodname.ElasticNet}ParaStore','data'),
    State(f'{methodname.Sarima}ParaStore','data'),
    State(f'{methodname.Sarimax}ParaStore','data'),
    State(f'{methodname.XGB_RF}ParaStore','data'),
    State(f'{methodname.GB}ParaStore','data'),
    State('exoTestStore', 'data'),
    State('productChoice','modified_timestamp'),
    State('driverChoice', 'modified_timestamp'),
    State('predictionsStore', 'modified_timestamp'),
    prevent_initial_call = True
)
def show_param(clicks, HW, Lasso, mlr, ridge, net, Sarima, Sarimax, XGB_RF, GB,
               exoTest, prod_time, drive_time, pred_time):
    if (clicks==0 or prod_time is None or drive_time is None or pred_time is None
        or (prod_time > pred_time) or (drive_time > pred_time)):
        raise PreventUpdate
    
    exo_test = pd.DataFrame.from_dict(exoTest.get('data'))
    exo_test.set_index('Date', inplace=True)
    exo_test.index = pd.to_datetime(exo_test.index, format='%d-%m-%Y')
    cat = exo_test.columns
    if not HW is None and not HW.get('data') is None and HW.get('data'):
        HW_c = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(f'Parameterwerte von {methodname.Holt_Winters} : '),
                            dash_table.DataTable(HW.get('data'), 
                                                    id=f'Parameter_{methodname.Holt_Winters}',)
                        ], width =4),
                ],
                    ),
        ]
    else: 
        HW_c = [html.Div(f'Keine Daten für {methodname.Holt_Winters} vorhanden.')]
    if not Lasso is None and not Lasso.get('data') is None and Lasso.get('data'):
        lasso_koeff = pd.DataFrame.from_dict(Lasso.get('data'))
        lasso_koeff.set_index('Koeffizienten', inplace=True)
        lasso_impact = exo_test.astype(float)*lasso_koeff.loc[cat, methodname.LASSO]
        lasso_impact['Intercept'] = lasso_koeff.loc['Intercept', methodname.LASSO]
        lasso_fig = px.bar(lasso_impact, y=lasso_impact.columns)
        lasso_fig.update_layout(
            title_text=f'Beiträge der Treiber zur Prognose für {methodname.LASSO}', legend_title_text='', 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        if exo_test.shape[0] < 20:
            lasso_fig.update_xaxes(nticks=exo_test.shape[0])
        lasso_graph = dcc.Graph(figure=lasso_fig)
        lasso_c = [
            dbc.Row(
                [lasso_graph]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(f'Parameterwerte von {methodname.LASSO}: '),
                            dash_table.DataTable(Lasso.get('data'), 
                                                    id = f'Parameter_{methodname.LASSO}', )
                            ], width =4),
                ],
                ),
        ]
    else: 
        lasso_c = [html.Div(f'Keine Daten für {methodname.LASSO} vorhanden.')]
    if not mlr is None and not mlr.get('data') is None and mlr.get('data'):
        mlr_koeff = pd.DataFrame.from_dict(mlr.get('data'))
        mlr_koeff.set_index('Koeffizienten', inplace=True)
        mlr_impact = exo_test.astype(float)*mlr_koeff.loc[cat, methodname.mlr]
        mlr_impact['Intercept'] = mlr_koeff.loc['Intercept', methodname.mlr]
        mlr_fig = px.bar(mlr_impact, y=mlr_impact.columns)
        mlr_fig.update_layout(
            title_text=f'Beiträge der Treiber zur Prognose für {methodname.mlr}', legend_title_text='', 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        if exo_test.shape[0] < 20:
            mlr_fig.update_xaxes(nticks=exo_test.shape[0])
        mlr_graph = dcc.Graph(figure=mlr_fig)
        mlr_c = [
            dbc.Row([mlr_graph]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(f'Parameterwerte von {methodname.mlr}: '),
                            dash_table.DataTable(mlr.get('data'), 
                                                    id = f'Parameter_{methodname.mlr}', )
                            ], width =4),
                ],
                ),
        ]
    else: 
        mlr_c = [html.Div(f'Keine Daten für {methodname.mlr} vorhanden.')]
    if not ridge is None and not ridge.get('data') is None and ridge.get('data'):
        ridge_koeff = pd.DataFrame.from_dict(ridge.get('data'))
        ridge_koeff.set_index('Koeffizienten', inplace=True)
        ridge_impact = exo_test.astype(float)*ridge_koeff.loc[cat, methodname.ridge]
        ridge_impact['Intercept'] = ridge_koeff.loc['Intercept', methodname.ridge]
        ridge_fig = px.bar(ridge_impact, y=ridge_impact.columns)
        ridge_fig.update_layout(
            title_text=f'Beiträge der Treiber zur Prognose für {methodname.ridge}', legend_title_text='', 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        if exo_test.shape[0] < 20:
            ridge_fig.update_xaxes(nticks=exo_test.shape[0])
        ridge_graph = dcc.Graph(figure=ridge_fig)
        ridge_c = [
            dbc.Row([ridge_graph]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(f'Parameterwerte von {methodname.ridge}: '),
                            dash_table.DataTable(ridge.get('data'), 
                                                    id = f'Parameter_{methodname.ridge}', )
                            ], width =4),
                ],
                ),
        ]
    else: 
        ridge_c = [html.Div(f'Keine Daten für {methodname.ridge} vorhanden.')]
    if not net is None and not net.get('data') is None and net.get('data'):
        net_koeff = pd.DataFrame.from_dict(net.get('data'))
        net_koeff.set_index('Koeffizienten', inplace=True)
        net_impact = exo_test.astype(float)*net_koeff.loc[cat, methodname.ElasticNet]
        net_impact['Intercept'] = net_koeff.loc['Intercept', methodname.ElasticNet]
        net_fig = px.bar(net_impact, y=net_impact.columns)
        net_fig.update_layout(
            title_text=f'Beiträge der Treiber zur Prognose für {methodname.ElasticNet}', legend_title_text='', 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        if exo_test.shape[0] < 20:
            net_fig.update_xaxes(nticks=exo_test.shape[0])
        net_graph = dcc.Graph(figure=net_fig)
        net_c = [
            dbc.Row([net_graph]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(f'Parameterwerte von {methodname.ElasticNet}: '),
                            dash_table.DataTable(net.get('data'), 
                                                    id = f'Parameter_{methodname.ElasticNet}', )
                            ], width =4),
                ],
                ),
        ]
    else: 
        net_c = [html.Div(f'Keine Daten für {methodname.ElasticNet} vorhanden.')]
    if not Sarima is None and not Sarima.get('data') is None and Sarima.get('data'):
        sarima_c = [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(f'Parameterwerte von {methodname.Sarima} : '),
                                dash_table.DataTable(Sarima.get('data'), 
                                                        id=f'Parameter_{methodname.Sarima}', )
                            ], width =4),
                    ],
                        ),
        ]
    else: 
        sarima_c = [html.Div(f'Keine Daten für {methodname.Sarima} vorhanden.')]
    if not Sarimax is None and not Sarimax.get('data') is None and Sarimax.get('data'):
        sarimax_koeff = pd.DataFrame.from_dict(Sarimax.get('data'))
        sarimax_koeff.set_index('Parameter', inplace=True)
        sarimax_impact = exo_test.astype(float)*sarimax_koeff.loc[cat, methodname.Sarimax].astype(float)
        if 'intercept' in sarimax_koeff.index:
            sarimax_impact['Intercept'] = sarimax_koeff.loc['intercept', methodname.Sarimax]
        sarimax_fig = px.bar(sarimax_impact, y=sarimax_impact.columns)
        sarimax_fig.update_layout(
            title_text=f'Beiträge der Treiber zur Prognose für {methodname.Sarimax}', legend_title_text='', 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        if exo_test.shape[0] < 20:
            sarimax_fig.update_xaxes(nticks=exo_test.shape[0])
        sarimax_graph = dcc.Graph(figure=sarimax_fig)
        sarimax_c = [
                dbc.Row([sarimax_graph]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(f'Parameterwerte von {methodname.Sarimax} : '),
                                dash_table.DataTable(Sarimax.get('data'), 
                                                        id=f'Parameter_{methodname.Sarimax}', )
                            ], width =4),
                    ],
                ),
        ]
    else: 
        sarimax_c = [html.Div(f'Keine Daten für {methodname.Sarimax} vorhanden.')]
    if not XGB_RF is None and not XGB_RF.get('data') is None and XGB_RF.get('data'):
        xgbrf_koeff = pd.DataFrame.from_dict(XGB_RF.get('data'))
        xgbrf_koeff.set_index('exogene Variablen', inplace=True)
        xgbrf_koeff.sort_values(methodname.XGB_RF, ascending=False, inplace=True)
        xgbrf_fig = px.bar(xgbrf_koeff, y=xgbrf_koeff.columns)
        xgbrf_fig.update_layout(
            title_text=f'Gewichtung der Treiber durch {methodname.XGB_RF}', showlegend=False, 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        xgbrf_fig.update_xaxes(nticks=exo_test.shape[1])
        xgbrf_graph = dcc.Graph(figure=xgbrf_fig)
        xgbrf_c = [
                dbc.Row([xgbrf_graph]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(f'Gewichtung der Treiber durch {methodname.XGB_RF}: '),
                                dash_table.DataTable(XGB_RF.get('data'), 
                                                        id=f'Parameter_{methodname.XGB_RF}', )
                            ], width =4),
                    ],
                ),
        ]
    else: 
        xgbrf_c = [html.Div(f'Keine Daten für {methodname.XGB_RF} vorhanden.')]
    if not GB is None and not GB.get('data') is None and GB.get('data'):
        gb_koeff = pd.DataFrame.from_dict(GB.get('data'))
        gb_koeff.set_index('exogene Variablen', inplace=True)
        gb_koeff.sort_values(methodname.GB, ascending=False, inplace=True)
        gb_fig = px.bar(gb_koeff, y=gb_koeff.columns)
        gb_fig.update_layout(
            title_text=f'Gewichtung der Treiber durch {methodname.GB}', showlegend=False, 
            xaxis_title_text='', yaxis_title_text='', yaxis_tickformat=',~r'
        )
        gb_fig.update_xaxes(nticks=exo_test.shape[1])
        gb_graph = dcc.Graph(figure=gb_fig)
        gb_c = [
                dbc.Row([gb_graph]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(f'Gewichtung der Treiber durch {methodname.GB}: '),
                                dash_table.DataTable(GB.get('data'), 
                                                        id=f'Parameter_{methodname.GB}', )
                            ], width =4),
                    ],
                ),
        ]
    else: 
        gb_c = [html.Div(f'Keine Daten für {methodname.GB} vorhanden.')]
    return HW_c, lasso_c, mlr_c, ridge_c, net_c, sarima_c, sarimax_c, xgbrf_c, gb_c

@callback(
    Output('cancel-button', 'active'),
    Input('cancel-button', 'n_clicks'),
    prevent_initial_call=True
)
def force_stop(clicks):
    if clicks > 0:
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)
        
