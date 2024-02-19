import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima import decompose
from statsmodels.tsa.seasonal import STL
from dash import dcc, html, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
import base64
import datetime
import io

dash.register_page(__name__, path='/Produktauswahl', name='Produktauswahl', order = 3) 


#To-Do: prevent callback trigger if you only click on the product page
layout = html.Div(id='productContainer', 
                  children=[
                        html.P("Wählen Sie die zu prognostizierende Variable aus:"),
                        dcc.Dropdown(
                            id='dropdown',
                            options=[],
                            value=None,
                            ),
                        dcc.Graph(id='product-graph'), 
                        html.P('',style={'color':'tomato'},id='warning-message'), 
                        html.Label('Visualisierung für Trend-Zerlegung und potentielle Ausreißer*.'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Verwendetes Verfahren: ')
                            ], width=3),
                            dbc.Col([
                                dbc.RadioItems(options={'MA':'Moving Averages', 'STL': 'STL'}, value='MA',
                                               id='dec_display_selection', inline=True)
                            ])
                        ]),
                        html.Div(id='arima_container', children=[
                            html.Div(), html.Div(),
                            ], hidden=True),
                        html.Div(id='stl_container', children=[
                            html.Div(), html.Div()
                            ], hidden=True),       
                        html.Hr(style={'visibility':'hidden'}),
                        dcc.Markdown(
                            '''
                            \* Hier zerlegen wir die Daten in Komponenten, die verschiedene Effekte  
                            repräsentieren. Die Summe der drei Komponenten ergibt das Gesamtprodukt.  
                            Die beiden Verfahren resultieren in unterschiedlichen Zerlegungen.  
                            
                            Der obere Graph zeigt, wie weit die einzelnen Datenpunkte vom Durchschnitt  
                            abweichen, gemessen in Vielfachen der Standardabweichung.  
                            Werte über 3 oder unter -3 (oder anderweitig auffällig) können bedeuten,  
                            dass es sich um einen Ausreißer handelt, also potentiell ein Fehler oder  
                            außergewöhnlicher Effekt.  
                            
                            Extreme Effekte können diese Darstellung dominieren, um Ausreißer in  
                            anderen Datenbereichen zu sehen, ggf. den Zeitraum einschränken.  
                            ''',
                            style={'fontStyle':'italic'}
                            ), 
                  ])

@callback(
    Output('dropdown', 'options'),
    Input('productStore', 'data'),
)
def update_options(productStore):
    if not productStore is None:
        df = pd.DataFrame.from_dict(productStore.get('data'))
        lst = df.columns.to_list()
        lst.remove('Date')
        return lst
    else:
        return []

@callback(
    Output('productContainer', 'children'),
    Output('productChoice', 'data'),
    Output('warning-message', 'children'),
    Output('arima_container', 'children'),
    Output('stl_container', 'children'),
    State('productStore', 'data'),
    State('trainingDateStore', 'data'),
    State('forecastDateStore', 'data'),
    State('productContainer', 'children'),
    Input('dropdown', 'value')
    )

def update_product_selection(productStore, Datem, Datec, children, choice):
    if not productStore is None and not choice is None:
        if not choice in productStore.get('data')[0].keys():
            return children, {}, '', [], []
        selection = {
            "data" : choice
        }
        Date_min = Datem.get('data')
        Date_cut = Datec.get('data')
        df = pd.DataFrame.from_dict(productStore.get('data'))
        df.index = df.Date
        df.Date=pd.to_datetime(df.Date, format="%d-%m-%Y")
        df_date_range = df[((pd.to_datetime(df.index, format="%d-%m-%Y") >= pd.to_datetime(Date_min, format="%d-%m-%Y"))&
                                    (pd.to_datetime(df.index, format="%d-%m-%Y") < pd.to_datetime(Date_cut, format="%d-%m-%Y")))]
        graph = px.line(df_date_range, x = df_date_range['Date'], y = df_date_range[choice])
        graph.update_xaxes(title_text='Datum')
        graph.update_layout(yaxis_tickformat = ',~r')
        children[2] = dcc.Graph(figure=graph)
        msg = 'Es fehlen Daten im Trainingszeitraum!' if df_date_range[choice].isna().sum() > 0 else ''
        
        pmd_children = []
        _, trend_ar, seas_ar, res_ar = decompose(df_date_range[choice], type_='additive', m=12)
        ar_z_score = (res_ar - res_ar.mean())/res_ar.std()
        ar_z_fig = px.line(x=df_date_range['Date'], y=ar_z_score)
        ar_z_fig.add_hline(y=3, line_color='red')
        ar_z_fig.add_hline(y=-3, line_color='red')
        ar_z_fig.update_layout(
            title_text='Wahrscheinlichkeit für Ausreißer', xaxis_title_text='Datum',
            yaxis_title_text='Standardabweichungen'
            )
        pmd_children.append(dcc.Graph(figure=ar_z_fig))
        pmd_fig = make_subplots(rows=3, cols=1)
        pmd_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=trend_ar,
                mode='markers+lines',
                name='Trend', 
            ), row=1, col=1
        )
        pmd_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=seas_ar,
                mode='markers+lines',
                name='Wiederkehrende Effekte',     
            ), row=2, col=1
        )
        pmd_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=res_ar,
                mode='markers+lines',
                name='Andere Effekte',    
            ), row=3, col=1
        )
        pmd_fig.update_layout(
            yaxis_tickformat = ',~r', yaxis2_tickformat = ',~r', 
            yaxis3_tickformat = ',~r', height=600,
            title_text = 'Zerlegung in Trend, zyklische Effekte und Rest'
            )
        pmd_children.append(dcc.Graph(figure=pmd_fig))
        
        stl_children = []
        stl_model = STL(df_date_range[choice], period=12)
        stl_dec = stl_model.fit()
        stl_z_score = (stl_dec.resid - stl_dec.resid.mean())/stl_dec.resid.std()
        stl_z_fig = px.line(x=df_date_range['Date'], y=stl_z_score)
        stl_z_fig.add_hline(y=3, line_color='red')
        stl_z_fig.add_hline(y=-3, line_color='red')
        stl_z_fig.update_layout(
            title_text='Wahrscheinlichkeit für Ausreißer', xaxis_title_text='Datum',
            yaxis_title_text='Standardabweichungen'
            )
        stl_children.append(dcc.Graph(figure=stl_z_fig))
        stl_fig = make_subplots(rows=3, cols=1)
        stl_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=stl_dec.trend,
                mode='markers+lines',
                name='Trend', 
            ), row=1, col=1
        )
        stl_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=stl_dec.seasonal,
                mode='markers+lines',
                name='Wiederkehrende Effekte',     
            ), row=2, col=1
        )
        stl_fig.add_trace(
            go.Scatter(
                x=df_date_range['Date'],
                y=stl_dec.resid,
                mode='markers+lines',
                name='Andere Effekte',    
            ), row=3, col=1
        )
        stl_fig.update_layout(
            yaxis_tickformat = ',~r', yaxis2_tickformat = ',~r', 
            yaxis3_tickformat = ',~r', height=600, 
            title_text = 'Zerlegung in Trend, zyklische Effekte und Rest'
            )
        stl_children.append(dcc.Graph(figure=stl_fig))
        return children, selection, msg, pmd_children, stl_children
    else:
        return children, {}, '', [], []
    
@callback(
    Output('arima_container', 'hidden'),
    Output('stl_container', 'hidden'),
    Input('dec_display_selection', 'value')
)
def toggle_display(choice):
    if choice is None:
        return no_update, no_update
    return 'STL' in choice, 'MA' in choice

@callback(
    Output('dropdown', 'value'),
    Input('productChoice', 'modified_timestamp'),
    State('productChoice', 'data')
)
def on_product(timeStamp, data):
    if timeStamp is None:
        raise PreventUpdate

    data = data or {}

    return data.get('data')

@callback(
    Output('WeiterProduktauswahl', 'data'),
    Output('WeiterTreiberauswahl', 'data', allow_duplicate=True),
    Input('dropdown', 'value'),
    State('warning-message', 'children'),
    prevent_initial_call = True
)
def toggle_button(selection, msg):
    result =  {
            'data': 'True'
    }
    if selection is None or not selection:
        return {}, {}
    if msg: 
        return {}, {}
    return result, no_update
