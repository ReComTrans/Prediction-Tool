import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
from dash import dcc, html, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from datetime import date

units_list = ['Euro €', 'Stückzahl Stk', 'einheitslos']

dash.register_page(__name__, path='/Voreinstellungen', name='Voreinstellungen', order=2) # '/' is home page

LABEL_WIDTH = 5
INPUT_WIDTH = 3

layout = html.Div(
    id = "presettings",
    children = [   
         dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Ab welchem Datum soll trainiert werden? *')
                    ],
                    width = LABEL_WIDTH,
                ),
                dbc.Col(
                    [
                        dcc.DatePickerSingle(
                            id = "trainingDate",
                            placeholder = 'TT/MM/JJJJ', 
                            display_format = 'DD/MM/YYYY' 
                        )
                    ],
                )             
            ],
            align='center',
        ),
        html.Div(id='warn_msg_train'),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Ab welchem Datum soll prognostiziert werden?')
                    ],
                    width = LABEL_WIDTH
                ),
                dbc.Col(
                    [
                        dcc.DatePickerSingle(
                            id = "forecastDate",
                            placeholder = 'TT/MM/JJJJ', 
                            display_format = 'DD/MM/YYYY'    
                        )
                    ],
                )             
            ],
            align='center',
        ),
        html.Div(id='warn_msg_pred'),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Welchen Zyklus/Periode haben die Daten? **')
                    ],
                    width = LABEL_WIDTH,
                ),
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id = 'seasonalityPicker', placeholder='Anzahl eingeben ... ', type= 'number', min=1)
                            ], xs=8, sm=6, md=5, lg=4, xl=4, xxl=4),
                            dbc.Col([
                                dbc.Label('Einheiten', id='label-season')
                            ], xs=4, sm=6, md=7, lg=8, xl=8, xxl=8)
                        ])
                        
                    ],
                    xs=7, sm=7, md=6, lg=5, xl=4, xxl=3
                )
            ],
            align='center',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Wie viele Zeitschritte sollen prognostiziert werden?')
                    ],
                    width = LABEL_WIDTH
                ),
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id = 'forecastingLabelWidth', placeholder='Zeitschritte eingeben ... ', type= 'number', min=1)
                            ], width=8),
                            dbc.Col([
                                dbc.Label('Einheiten', id='label-steps')
                            ], width=4)                            
                        ])
                    ],
                    xs=7, sm=7, md=6, lg=5, xl=4, xxl=3
                )             
            ],
            align='center',
        ),
        html.Div(id='warn_msg_lof'),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Welche Einheit besitzen die Prognosewerte? ***')
                    ],
                    width = LABEL_WIDTH
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(id="units", options= units_list )
                    ],
                    xs=7, sm=7, md=6, lg=5, xl=4, xxl=3
                )             
            ],
            align='center',
        ),
        html.Div(id='warn_msg_unit'),
        html.Hr(
            style={'margin-bottom':'20px',}
            ),
        dcc.Markdown(
            '''
            \* Sollte deutlich vor dem Prognosestart liegen, idealerweise  
            mehrere Jahre (bei Monatsdaten) oder Monate (bei Tagesdaten),  
            minimaler Abstand abhängig von der Anwendung. 
            
            \*\* Länge des dominanten Zyklus, periodischen oder saisonalen Effekts.  
            Beschreibt den Rythmus des wichtigsten wiederkehrenden Effekts in den Daten.  
            Beispiel: Jahreszeiten-Effekt &rarr; Zyklus = 12 Monate,  
            oder: Wochentagseffekt &rarr; Zyklus = 7 Tage  
            Falls unbekannt oder falls kein periodisches Verhalten existiert: Mit default testen.  
            
            \*\*\* Legt das Rundungsverhalten fest.  
            ''',
            style={'fontStyle':'italic'}
        )
    ],
    className="gy-5"
     
)


#### trainingDate callbacks
@callback(
    Output('trainingDate', 'date'),
    Input('trainingDateStore', 'modified_timestamp'),
    State('trainingDateStore', 'data')
)
def on_trainingDate(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate
    
    data = data or {}
    date = data.get('data')
    if not date is None: 
        date = pd.to_datetime(date, format="%d-%m-%Y")
    return date

@callback(
    Output('trainingDateStore', 'data'),
    Output('warn_msg_train', 'children'),
    Input('trainingDate', 'date'),
)
def update_trainingDateStore(date):
    if not date is None:
        date = pd.to_datetime(date, format="ISO8601").strftime(format="%d-%m-%Y")
    store = {
        "data": date,
    }
    if date is None:
        msg = dbc.Row([
                    dbc.Col([],width=LABEL_WIDTH),
                    dbc.Col([html.Label('Trainingsdatum fehlt oder ist inkorrekt (Format: TT/MM/JJJJ).', 
                                        style={'color': 'tomato'})])
              ]),
    else:
        msg = []
    return store, msg

@callback(
    Output('trainingDate', 'min_date_allowed'),
    Output('trainingDate', 'max_date_allowed'),
    Input('productStore', 'data'),
    State('granularityStore', 'data')
)
def update_trainingDatePicker_options(store, gran):
    date_min = pd.to_datetime(store['data'][0]['Date'], format="%d-%m-%Y")
    date_max = (pd.to_datetime(store['data'][-7]['Date'], format="%d-%m-%Y") if gran['data']=='Tage'
                else pd.to_datetime(store['data'][-12]['Date'], format="%d-%m-%Y"))
    return date_min, date_max

#### forecast callbacks
@callback(
    Output('forecastDate', 'date'),
    Input('forecastDateStore', 'modified_timestamp'),
    State('forecastDateStore', 'data')
)
def on_forecastDate(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate
    
    data = data or {}
    date = data.get('data')
    if not date is None: 
        date = pd.to_datetime(date, format="%d-%m-%Y")
    return date

@callback(
    Output('forecastDateStore', 'data'),
    Output('warn_msg_pred', 'children'),
    Input('forecastDate', 'date')
)
def update_ForecastDateStore(date):
    if not date is None:
        date = pd.to_datetime(date, format="ISO8601").strftime(format="%d-%m-%Y")
    store = {
        "data": date,
    }
    if date is None: 
        msg = dbc.Row([
                    dbc.Col([], width=LABEL_WIDTH),
                    dbc.Col([html.Label('Prognosedatum fehlt oder ist inkorrekt (Format: TT/MM/JJJJ).', style={'color': 'tomato'})]),            
              ]),
    else: 
        msg = []
    return store, msg

@callback(
    Output('forecastDate', 'min_date_allowed'),
    Output('forecastDate', 'max_date_allowed'),
    Output('forecastDate', 'initial_visible_month'),
    Input('productStore', 'data'),
    State('granularityStore', 'data')
)
def update_forecastDatePicker_options(store, gran):
    date_min = (pd.to_datetime(store['data'][7]['Date'], format="%d-%m-%Y") if gran['data']=='Tage'
                else pd.to_datetime(store['data'][12]['Date'], format="%d-%m-%Y"))
    date_max = pd.to_datetime(store['data'][-1]['Date'], format="%d-%m-%Y")
    if gran['data'] == 'Tage':
        date_max += pd.Timedelta(1, 'D')
    elif gran['data'] == 'Monate':
        if date_max.month in [1,3,5,7,8,10,12]:
            date_max += pd.Timedelta(31, 'D')
        elif date_max.month in [4,6,9,11]:
            date_max += pd.Timedelta(30, 'D')
        else: 
            date_max += pd.Timedelta(28, 'D') if date_max.year%4 else pd.Timedelta(29, 'D')
    else: 
        return no_update, no_update, no_update
    return date_min, date_max, date_max

#### seasonality callbacks

@callback(
    Output('seasonalityPicker', 'value'),
    Output('label-season', 'children'),
    Output('label-steps', 'children'),
    Input('seasonalityStore', 'modified_timestamp'),
    State('seasonalityStore', 'data'),
    Input('granularityStore', 'modified_timestamp'),
    State('granularityStore', 'data'),
)
def on_seasonalityPicker(seas_time, seas_data, gran_time, gran_data):
    if seas_time is None and gran_time is None: 
        raise PreventUpdate
    seas_data = seas_data or {}
    seas = seas_data.get('data')
    if not seas is None and seas: 
        if gran_data.get('data')=='Tage':
            num = 7 
            msg = 'Tage (default wöchentlich)'
        else: 
            num = 12
            msg = 'Monate (default jährlich)'
        if gran_time > seas_time: 
            return num, msg, gran_data.get('data')
        return seas, msg, gran_data.get('data')
    if not gran_data.get('data') is None:
        if gran_data.get('data')=='Tage':
            num = 7 
            msg = 'Tage (default wöchentlich)'
        else: 
            num = 12
            msg = 'Monate (default jährlich)'
        return num, msg, gran_data.get('data')
    return no_update, no_update, no_update

@callback(
    Output('seasonalityStore', 'data'),
    Input('seasonalityPicker', 'value')
)
def update_seasonalityStore(value):
    store = {
        "data": value
    }
    return store

#### forecastingLabelWidth callbacks

@callback(
    Output('forecastingLabelWidth', 'value'),
    Input('forecastingLabelWidthStore', 'modified_timestamp'),
    State('forecastingLabelWidthStore', 'data')
)
def on_forecastingLabelWidth(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate
    
    data = data or {}

    return data.get('data')

@callback(
    Output('forecastingLabelWidthStore', 'data'),
    Output('warn_msg_lof', 'children'),
    Input('forecastingLabelWidth', 'value'),
)
def update_forecastingLabelWidthStore(value):
    store = {
        "data" : value
    }
    if value is None:
        msg = dbc.Row([
                    dbc.Col([], width=LABEL_WIDTH),
                    dbc.Col([html.Label('Keine valide Prognoselänge angegeben.', style={'color': 'tomato'})]),
              ]),
    else: 
        msg = []
    return store, msg


#### units callbacks

@callback(
    Output('units', 'value'),
    Input('unitsStore', 'modified_timestamp'),
    State('unitsStore', 'data')
)
def on_units(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate
    
    data = data or {}

    return data.get('data')

@callback(
    Output('unitsStore', 'data'),
    Output('warn_msg_unit', 'children'),
    Input('units', 'value'),
)
def update_unitsStore(value):
    store = {
        "data" : value
    }
    if value is None:
        msg = dbc.Row([
                    dbc.Col([], width=LABEL_WIDTH),
                    dbc.Col([html.Label('Keine Einheit ausgewählt.', style={'color': 'tomato'})]),
              ]),
    else: 
        msg = []
    return store, msg

@callback(
    Output('WeiterVoreinstellungen', 'data'),   
    Output('WeiterProduktauswahl', 'data', allow_duplicate=True),
    Output('WeiterTreiberauswahl', 'data', allow_duplicate=True),
    Input('trainingDateStore', 'data'),
    Input('forecastDateStore', 'data'),
    Input('forecastingLabelWidthStore', 'data'),
    Input('unitsStore', 'data'),
    State('granularityStore', 'data'),
    prevent_initial_call = True
)
def toggle_button(train, forecast, labelWidth, unit, gran):
    result =  {
            'data': 'True'
    }
    for store in [train, forecast, labelWidth, unit]:
        if store is None or not store or store.get('data') is None:
            return {}, {}, {}
    gran = gran.get('data')
    limit = pd.Timedelta(7, 'D') if gran=='Tage' else pd.Timedelta(365, 'D')
    if (
        pd.to_datetime(forecast.get('data'), format="%d-%m-%Y") - 
        pd.to_datetime(train.get('data'), format="%d-%m-%Y") < limit
    ):
        return {}, {}, {}
    return result, no_update, no_update


