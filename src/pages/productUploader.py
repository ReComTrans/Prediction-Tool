import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import dcc, html, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
import base64
import datetime
import io

from dataPreparation.Data import load_dummy_data, load_dummy_drivers

dash.register_page(__name__, path='/', name='Uploader', order = 1) 

layout = html.Div([
    dbc.Label('Bitte Produkt- und Treiberdaten hochladen als Excel oder CSV.'), 
    html.P('Die erste Spalte muss eine Datumsspalte sein im Format TT-MM-JJJJ oder TT.MM.JJJJ.'),
    html.Hr(),
    html.P('Produktdaten hochladen:'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div(id='product_feedback'),
    html.Hr(), 
    html.P('Treiberdaten hochladen:'),
    dcc.Upload(
        id='upload-driver',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div(id='driver_feedback'),
    html.Hr(),
    dbc.Button('Beispieldaten laden', id='dummy-button', color='primary', className='me-1', n_clicks=0)
])

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        msg = html.Div(['There was an error processing this file.'])
        return msg, None, None
    nan_list = []
    df.rename(columns={df.columns[0] :'Date'}, inplace=True)
    df.dropna(axis='index', subset='Date', inplace=True)
    for c in df.columns[1:]:
        if 'Unnamed' in c:
            df.drop(c, axis='columns', inplace=True)
            continue
        if (df[c].map(type)==str).sum() > 0:
            msg = html.Div([f'In Spalte "{c}" ist an mindestens einer Stelle Text (statt einer Zahl)'])
            return msg, None, None
        nan_list.append(df[c].isna().sum())
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        df['Date'].dt.strftime('%d-%m-%Y')
    except Exception as e:
        print(e) 
        msg = html.Div(['Fehler beim Auslesen des Datums in der ersten Spalte.'])
        return msg, None, None
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df.replace(0, 0.0, inplace=True)

    first = df['Date'].iloc[0]
    second = df['Date'].iloc[1]
    last = df['Date'].iloc[-1]
    diff = (second - first) / pd.Timedelta(1, 'D')
    total = (last - first) / pd.Timedelta(1, 'D')
    if diff in [28,29,30,31]:
        gran = 'Monate'
        for i in range(len(df['Date'])-2):
            first = df['Date'][i+1]
            second = df['Date'][i+2]
            diff = (second - first) / pd.Timedelta(1, 'D')
            if not diff in [28,29,30,31]:
                df = html.Div([
                    f"Granularität Monate festgestellt aber der Abstand zwischen {df['Date'][i+1]} und {df['Date'][i+2]} ist {round(diff)} Tage und kein Monat."
                ])
                return df, no_update, no_update
    elif diff == 1:
        gran = 'Tage'
        for i in range(len(df['Date'])-2):
            first = df['Date'][i+1]
            second = df['Date'][i+2]
            diff = (second - first) / pd.Timedelta(1, 'D')
            if diff != 1:
                df = html.Div([
                    f"Granularität Tage festgestellt aber der Abstand zwischen {df['Date'][i+1]} und {df['Date'][i+2]} ist {round(diff)} Tage statt einem."
                ])
                return df, no_update, no_update
    else: 
        gran = None
        df = html.Div([
                f"Der Abstand zwischen {df['Date'][0]} und {df['Date'][1]} ist {round(diff)} Tage und weder ein Monat noch ein Tag."
            ])
        return df, no_update, no_update
    if gran == 'Monate':
        df['Date'] = [date.replace(day=1) for date in df['Date']]
    if (gran == 'Monate' and total < 366) or (gran == 'Tage' and total < 7):
        df = html.Div([f"Nicht genug Daten für eine Prognose! " + ("(mind. 1 Woche)" if gran=='Tage' else "(mind. 1 Jahr)")])
        return df, no_update, no_update
    df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
    df.index = df.Date.astype(str)
    return df, nan_list, gran

@callback(
        Output('productStore', 'data'),
        Output('driverStore', 'data', allow_duplicate=True),
        Output('granularityStore', 'data', allow_duplicate=True),
        Output('product_feedback', 'children'),
        Output('productChoice', 'data', allow_duplicate=True), 
        Output('driverChoice', 'data', allow_duplicate=True), 
        Output('trainingDateStore', 'data', allow_duplicate=True),
        Output('forecastDateStore', 'data', allow_duplicate=True),
        Output('forecastingLabelWidthStore', 'data', allow_duplicate=True),
        Output('unitsStore', 'data', allow_duplicate=True),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('productStore', 'data'),
        State('driverStore', 'data'),
        prevent_initial_call = True
        )
def update_product_uploader(contents, filename, store, dstore):
    if not contents is None:
        df, nan_list, gran = parse_data(contents, filename)
        if type(df) is html.Div:
            return no_update, no_update, no_update, df, no_update, no_update, no_update, no_update, no_update, no_update 
        if not dstore is None and len(dstore.keys())>0:
            driver = pd.DataFrame.from_dict(dstore.get('data'))
            driver.index = driver['Date']
            dind = pd.to_datetime(pd.Index(driver['Date']), format='%d-%m-%Y')
            pind = pd.to_datetime(df.index, format='%d-%m-%Y')
            driver = pd.DataFrame(driver, index=dind.join(pind, how='outer').to_series().dt.strftime('%d-%m-%Y'))
            driver['Date'] = driver.index 
            driver_dict = {
                'data': driver.to_dict('records')
            }
        else: 
            driver_dict = no_update
        dict = { 
            'data': df.to_dict('records')
            }
        gran_dict = {
            'data': gran
        }
        column_df = pd.DataFrame()
        column_df['Produkte'] = df.columns[1:]
        column_df['Fehlende Einträge'] = nan_list
        msg = html.Div([
            html.P(f"Produktdaten hochgeladen von {df.iloc[0]['Date']} bis {df.iloc[-1]['Date']} mit den folgenden {len(df.columns)-1} Spalten:"),
            dbc.Col(
                dash.dash_table.DataTable(
                    column_df.to_dict('records'),
                ),
                width=3)
        ])
        return  dict, driver_dict, gran_dict, msg, {}, {}, {}, {}, {}, {}
    elif not store is None and len(store.keys())>0:
        column_df = pd.DataFrame(list(store['data'][0].keys())[1:], columns=['Produkte'])
        msg = html.Div([
            html.P(f"Produktdaten hochgeladen von {store['data'][0]['Date']} bis {store['data'][-1]['Date']} mit den folgenden {len(store['data'][0].keys())-1} Spalten:"),
            dbc.Col(
                dash.dash_table.DataTable(
                    column_df.to_dict('records'),
                ),
            width=3)
        ])
        return store, no_update, no_update, msg, no_update, no_update, no_update, no_update, no_update, no_update
    else:
        return {}, {}, {}, [], no_update, no_update, no_update, no_update, no_update, no_update
    
@callback(
        Output('driverStore', 'data'),
        Output('driver_feedback', 'children'),
        Output('productChoice', 'data', allow_duplicate=True), 
        Output('driverChoice', 'data', allow_duplicate=True), 
        Output('trainingDateStore', 'data', allow_duplicate=True),
        Output('forecastDateStore', 'data', allow_duplicate=True),
        Input('upload-driver', 'contents'),
        State('upload-driver', 'filename'),
        State('driverStore', 'data'),
        State('productStore', 'data'),
        prevent_initial_call = True
        )
def update_driver_uploader(contents, filename, store, pstore):
    if not contents is None:
        df, nan_list, _ = parse_data(contents, filename)
        if type(df) is html.Div:
            return no_update, df, no_update, no_update, no_update, no_update 
        if not pstore is None and len(pstore.keys())>0:
            prod = pd.DataFrame.from_dict(pstore.get('data'))
            pind = pd.to_datetime(pd.Index(prod['Date']), format='%d-%m-%Y')
            dind = pd.to_datetime(df.index, format='%d-%m-%Y')
            df = pd.DataFrame(df, index=dind.join(pind, how='outer').to_series().dt.strftime('%d-%m-%Y'))
            df['Date'] = df.index 
        dict = { 
            'data' : df.to_dict('records')
            }
        column_df = pd.DataFrame()
        column_df['Treiber'] = df.columns[1:]
        column_df['Fehlende Einträge'] = nan_list
        msg = html.Div([
            html.P(f"Treiberdaten hochgeladen von {df.iloc[0]['Date']} bis {df.iloc[-1]['Date']} mit den folgenden {len(df.columns)-1} Spalten:"),
            dbc.Col(
                dash.dash_table.DataTable(
                    column_df.to_dict('records'),
                ),
                width=3)
        ])
        return  dict, msg, {}, {}, {}, {}
    elif not store is None and len(store.keys())>0:
        column_df = pd.DataFrame(list(store['data'][0].keys())[1:], columns=['Treiber'])
        msg = html.Div([
            html.P(f"Treiberdaten hochgeladen von {store['data'][0]['Date']} bis {store['data'][-1]['Date']} mit den folgenden {len(store['data'][0].keys())-1} Spalten:"),
            dbc.Col(
                dash.dash_table.DataTable(
                    column_df.to_dict('records'),
                ),
            width=3)
        ])
        return store, msg, no_update, no_update, no_update, no_update
    else:
        return {}, [], no_update, no_update, no_update, no_update
    
@callback(
    Output('productStore', 'data', allow_duplicate=True),
    Output('product_feedback', 'children', allow_duplicate=True),
    Output('driverStore', 'data', allow_duplicate=True),
    Output('driver_feedback', 'children', allow_duplicate=True),
    Output('granularityStore', 'data', allow_duplicate=True),
    Output('dummy-button', 'n_clicks'),
    Output('productChoice', 'data', allow_duplicate=True), 
    Output('driverChoice', 'data', allow_duplicate=True), 
    Output('trainingDateStore', 'data', allow_duplicate=True),
    Output('forecastDateStore', 'data', allow_duplicate=True),
    Output('forecastingLabelWidthStore', 'data', allow_duplicate=True),
    Output('unitsStore', 'data', allow_duplicate=True),
    Input('dummy-button', 'n_clicks'),
    prevent_initial_call=True
)
def upload_dummy_data(clicks):
    if clicks==0:
        raise PreventUpdate
    product = load_dummy_data()
    prod_dict = { 
        'data' : product.to_dict('records')
        }
    prod_columns = pd.DataFrame()
    prod_columns['Produkte'] = product.columns[1:]
    prod_msg = html.Div([
        html.P(f"Produktdaten hochgeladen von {product.iloc[0]['Date']} bis {product.iloc[-1]['Date']} mit den folgenden {len(product.columns)-1} Spalten:"),
        dbc.Col(
            dash.dash_table.DataTable(
                prod_columns.to_dict('records'),
            ),
            width=3)
    ])
    driver = load_dummy_drivers()
    driver_dict = { 
        'data' : driver.to_dict('records')
        }
    driver_columns = pd.DataFrame()
    driver_columns['Treiber'] = driver.columns[1:]
    driver_msg = html.Div([
        html.P(f"Treiberdaten hochgeladen von {driver.iloc[0]['Date']} bis {driver.iloc[-1]['Date']} mit den folgenden {len(driver.columns)-1} Spalten:"),
        dbc.Col(
            dash.dash_table.DataTable(
                driver_columns.to_dict('records'),
            ),
            width=3)
    ])
    return prod_dict, prod_msg, driver_dict, driver_msg, {'data': 'Monate'}, 0, {}, {}, {}, {}, {}, {}

@callback(
    Output('WeiterUploader', 'data'),   
    Output('WeiterVoreinstellungen', 'data', allow_duplicate=True),   
    Output('WeiterProduktauswahl', 'data', allow_duplicate=True),   
    Output('WeiterTreiberauswahl', 'data', allow_duplicate=True),   
    Input('productStore', 'data'),
    Input('driverStore', 'data'), 
    prevent_initial_call = True
)
def toggle_button(p_store, d_store):
    result =  {
            'data': 'True'
    }
    if p_store is None or not p_store or p_store.get('data') is None:
        return {}, {}, {}, {}
    if d_store is None or not d_store or d_store.get('data') is None:
        return {}, {}, {}, {}
    return result, no_update, no_update, no_update
