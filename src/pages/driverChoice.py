import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima import decompose
from statsmodels.tsa.seasonal import STL
from dash import dcc, html, callback, Output, Input, State, no_update, ctx
from dash.exceptions import PreventUpdate
from datetime import date

dash.register_page(__name__, path='/Treiberauswahl', name='Treiberauswahl', order = 4) 

layout = html.Div(
    id = 'driverContainer',
    children=[
        html.Div(
            id='choiceContainer',
            hidden=False,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.P("Wählen Sie die Treiber aus (Minimum zwei):"),
                    ]
                    ), 
                    dbc.Col([
                        dbc.Row([
                            dbc.Switch(
                                id='selection-orientation', 
                                label='Treiber als Liste anzeigen', 
                                )
                        ]), 
                        dbc.Row([
                            dbc.Switch(
                                id='dropdown-display',
                                label='Durchsuchbares Eingabefeld'
                            )
                        ])
                    ]
                    )
                ]
                ),
                html.Div([                
                    dcc.Dropdown(options=[], id='driver-dropdown', multi=True,
                                 placeholder='Mehrere Treiber auswählen...'),
                ], id='driver-dropdown-container', hidden=True
                ),
                html.Hr(style={'visibility':'hidden'}),
                html.Div([
                    dbc.Checklist(options=[], id='driver-selection', inline=True),  
                ], id='driver-check-container', hidden=False
                ),
                html.Hr(),
            ]
            ),  
        dbc.Switch(label='Ausgewählte Treiber/ausgewähltes Produkt in einem Plot vergleichen (für drei und mehr ohne Skala)',
                      id='compare-display'),
        dbc.Switch(label='Potentielle Ausreißer anzeigen', id='outlier-display'),
        dbc.Switch(label='Korrelationsmatrix anzeigen', id='cm-display'), 
        html.Div(
            id='compareContainer',
            hidden=True,
            children=[
                html.Hr(),
                dbc.Checklist(options=[], id='compare-selection', inline=True),
            ]
            ),
        html.Div(
            id='outlierContainer',
            hidden=True, 
            children=[
                html.Hr(),
                dbc.RadioItems(options=[], id='outlier-selection', inline=True),
                html.Hr(),
                html.Label('Visualisierung für Trend-Zerlegung und potentielle Ausreißer*.'),
                dbc.Row([
                            dbc.Col([
                                dbc.Label('Verwendetes Verfahren: ')
                            ], width=3),
                            dbc.Col([
                                dbc.RadioItems(options={'MA':'Moving Averages', 'STL': 'STL'}, value='MA',
                                               id='outlier-mode-select', inline=True)
                            ])
                        ]),
            ]
            ),
        html.Div(id='missing-message'), # This will contain missing data warning 
        html.Div(), # This will contain correlation warning strings
        html.Div(), # This will contain correlation matrix plot
        dcc.Graph(), # This will contain the driver plot(s)
    ]
     
)

#### driver callbacks

@callback(
    Output('driver-selection', 'inline'),
    Output('driver-dropdown-container', 'hidden'),
    Output('driver-check-container', 'hidden'),
    Input('selection-orientation', 'value'),
    Input('dropdown-display', 'value'),
)
def change_select_layout(orient_choice, drop_choice):
    return not orient_choice, not drop_choice, drop_choice

@callback(
    Output('driver-selection', 'options'),
    Output('outlier-selection', 'options'),
    Output('driver-dropdown', 'options'),
    Input('driverStore', 'data'),
)
def update_options(driverStore):
    if not driverStore is None:
        df = pd.DataFrame.from_dict(driverStore.get('data'))
        lst = df.columns.to_list()
        lst.remove('Date')
        return lst, lst, lst
    else:
        return [], [], []

@callback(
    Output('outlier-display', 'value'),
    Input('compare-display', 'value')
)
def switch_compare(choice):
    return False if choice else no_update

@callback(
    Output('compare-display', 'value'),
    Input('outlier-display', 'value')
)
def switch_outlier(choice):
    return False if choice else no_update

@callback(
        Output('driverContainer', 'children'),
        Output('driverChoice', 'data'),
        Output('choiceContainer', 'hidden'),
        Output('compareContainer', 'hidden'),
        Output('outlierContainer', 'hidden'),
        Input('driver-selection', 'value'), 
        Input('driver-dropdown', 'value'),
        Input('compare-display', 'value'),
        Input('outlier-display', 'value'),
        Input('cm-display', 'value'),
        Input('compare-selection', 'value'),
        Input('outlier-selection', 'value'),
        Input('outlier-mode-select', 'value'),
        State('driverStore', 'data'),
        State('productStore', 'data'),
        State('productChoice','data'),
        State('driverContainer', 'children'), 
        State('trainingDateStore', 'data'), 
        State('forecastDateStore', 'data'),
        State('forecastingLabelWidthStore','data'),
)
def update_driver_selection(choice, dchoice, comp_display, outlier_display, cm_display, comp_choice, 
                            outlier_choice, outlier_mode, 
                            driverStore, productStore, pchoice, children, train_store, pred_store, pred_len):
    if ctx.triggered_id == 'driver-dropdown': 
        choice = dchoice
    store = {
        "data" : choice
    }
    if (choice is None or len(choice)==0 or driverStore is None) and not outlier_display: 
        return children, store, False, True, True
    for c in choice:
        if not c in driverStore.get('data')[0].keys():
            return children, {}, False, True, True
    
    driver = pd.DataFrame.from_dict(driverStore.get('data'))
    driver.set_index('Date', drop=False, inplace=True)
    driver['Date']=pd.to_datetime(driver['Date'], format="%d-%m-%Y")
    product = pd.DataFrame.from_dict(productStore.get('data'))
    product.set_index('Date', drop=False, inplace=True)
    product['Date']=pd.to_datetime(product['Date'], format="%d-%m-%Y")
        
    train_date = pd.to_datetime(train_store['data'], format="%d-%m-%Y")
    pred_date = pd.to_datetime(pred_store['data'], format="%d-%m-%Y")
    driver_timeframe = driver[(pd.to_datetime(driver.index, format="%d-%m-%Y") >= pd.to_datetime(train_date, format="%d-%m-%Y"))&
                                           (pd.to_datetime(driver.index, format="%d-%m-%Y") < pd.to_datetime(pred_date, format="%d-%m-%Y"))]
    product_timeframe = product[(pd.to_datetime(product.index, format="%d-%m-%Y") >= pd.to_datetime(train_date, format="%d-%m-%Y"))&
                                           (pd.to_datetime(product.index, format="%d-%m-%Y") < pd.to_datetime(pred_date, format="%d-%m-%Y"))]
    incomplete_drivers = [d for d in choice if driver_timeframe[d].isna().sum()>0]
    if len(incomplete_drivers) > 0:
        if len(incomplete_drivers) > 1:
            msg = 'Die Treiber '
            for d in incomplete_drivers[:-2]:
                msg += '\"' + d + '\", '
            msg += '\"' + incomplete_drivers[-2] + '\" und \"' + incomplete_drivers[-1] + '\" haben '
        else: 
            msg = 'Der Treiber \"' + incomplete_drivers[0] + '\" hat '
        msg += 'fehlende Einträge im gewählten Zeitraum!'
        children[-4]['props']['children'] = [html.Hr()]
        children[-4]['props']['children'].extend([html.P(msg, style={'color':'tomato'})])
    else: 
        children[-4]['props']['children'] = []
    temp = driver[(pd.to_datetime(driver.index, format="%d-%m-%Y") >= pd.to_datetime(pred_date, format="%d-%m-%Y"))]
    driver_timeframe = pd.concat([driver_timeframe,temp.loc[temp.index[0:pred_len['data']],:]])
    corr_df, cm = calc_correlations(driver_timeframe, choice, cutoff=0.6, return_correlation_matrix=True) 
    if len(corr_df) > 0 and not comp_display and not outlier_display:
        children[-3]['props']['children'] = [html.Hr()]
        children[-3]['props']['children'].extend([html.P(f'Achtung: {a} und {b} sind stark korreliert ({c})') 
                                                 for (a,b,c) in corr_df.values])
    else: 
        children[-3]['props']['children'] = []
    if cm_display:
        cm_fig = px.imshow(cm, width=200*np.sqrt(len(cm)), height=200*np.sqrt(len(cm)), 
                           color_continuous_scale='RdBu_r')
        children[-2] = dcc.Graph(figure=cm_fig)
    else:
        children[-2] = html.Div()   
    if comp_display and not comp_choice is None:  
        driver_product_timeframe=pd.concat([driver_timeframe, product_timeframe[pchoice.get('data')]], axis=1)
        if len(comp_choice) == 0:
            children[-1] = html.Div(
                                    [html.Hr(),
                                    html.P('Bitte Treiber für den Vergleich auswählen.')]
                                    )
        elif len(comp_choice) == 1:
            graph = go.Figure()
            graph.add_trace(
                go.Scatter(x = driver_timeframe['Date'], y = driver_product_timeframe[comp_choice[0]], name = comp_choice[0])
            )
            graph.update_xaxes(title_text="Datum")
            graph.update_yaxes(title_text=comp_choice[0])
            graph.update_layout(height = 450, yaxis_tickformat = ',~r')
            children[-1] = dcc.Graph(figure=graph)
        elif len(comp_choice) == 2:
            graph = make_subplots(specs=[[{"secondary_y": True}]])
            graph.add_trace(
                go.Scatter(x = driver_timeframe['Date'], y = driver_product_timeframe[comp_choice[0]], name = comp_choice[0]),
                secondary_y=False,
            )
            graph.add_trace(
                go.Scatter(x = driver_timeframe['Date'], y = driver_product_timeframe[comp_choice[1]], name = comp_choice[1]),
                secondary_y=True,
            )
            graph.update_xaxes(title_text="Datum")
            graph.update_yaxes(title_text=comp_choice[0], tickformat=',~r', secondary_y=False)
            graph.update_yaxes(title_text=comp_choice[1], tickformat=',~r', tickmode="sync", secondary_y=True)
            graph.update_layout(height = 450)
            
            children[-1] = dcc.Graph(figure=graph)
        else:
            graph = go.Figure(
                data=go.Scatter(x = driver_timeframe['Date'], y = driver_product_timeframe[comp_choice[0]], name = comp_choice[0])
            )
            for i in range(1,len(comp_choice)):
                graph.add_trace(
                    go.Scatter(x = driver_timeframe['Date'], y = driver_product_timeframe[comp_choice[i]], name = comp_choice[i], yaxis=f'y{i+1}'),
                )
            graph.update_xaxes(title_text="Datum")
            graph.update_yaxes(showticklabels=False,tickmode="sync")
            opts = {f'yaxis{i+1}':dict(anchor='free',overlaying='y',showticklabels=False,tickmode="sync") for i in range(1,len(comp_choice))}
            graph.update_layout(opts)
            graph.update_layout(height = 450, yaxis_tickformat = ',~r')
            children[-1] = dcc.Graph(figure=graph)
            
    if outlier_display and not outlier_choice is None:
        if not outlier_mode is None and 'MA' in outlier_mode:
            outlier_children = []
            _, trend_ar, seas_ar, res_ar = decompose(driver_timeframe[outlier_choice], type_='additive', m=12)
            ar_z_score = (res_ar - res_ar.mean())/res_ar.std()
            ar_z_fig = px.line(x=driver_timeframe['Date'], y=ar_z_score)
            ar_z_fig.add_hline(y=3, line_color='red')
            ar_z_fig.add_hline(y=-3, line_color='red')
            ar_z_fig.update_layout(
                title_text='Wahrscheinlichkeit für Ausreißer', xaxis_title_text='Datum',
                yaxis_title_text='Standardabweichungen'
                )
            outlier_children.append(dcc.Graph(figure=ar_z_fig))
            pmd_fig = make_subplots(rows=4, cols=1)
            pmd_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=trend_ar,
                    mode='markers+lines',
                    name='Trend', 
                ), row=1, col=1
            )
            pmd_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=seas_ar,
                    mode='markers+lines',
                    name='Wiederkehrende Effekte',     
                ), row=2, col=1
            )
            pmd_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=res_ar,
                    mode='markers+lines',
                    name='Andere Effekte',    
                ), row=3, col=1
            )
            pmd_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=driver_timeframe[outlier_choice],
                    mode='markers+lines',
                    name='Vollständiger Treiber',    
                ), row=4, col=1
            )
            pmd_fig.update_layout(
                yaxis_tickformat = ',~r', yaxis2_tickformat = ',~r', 
                yaxis3_tickformat = ',~r', yaxis4_tickformat = ',~r',
                height=800,
                title_text = 'Zerlegung in Trend, zyklische Effekte und Rest'
                )
            outlier_children.append(dcc.Graph(figure=pmd_fig))
            
        elif not outlier_mode is None and 'STL' in outlier_mode:
            outlier_children = []
            stl_model = STL(driver_timeframe[outlier_choice], period=12)
            stl_dec = stl_model.fit()
            stl_z_score = (stl_dec.resid - stl_dec.resid.mean())/stl_dec.resid.std()
            stl_z_fig = px.line(x=driver_timeframe['Date'], y=stl_z_score)
            stl_z_fig.add_hline(y=3, line_color='red')
            stl_z_fig.add_hline(y=-3, line_color='red')
            stl_z_fig.update_layout(
                title_text='Wahrscheinlichkeit für Ausreißer', xaxis_title_text='Datum',
                yaxis_title_text='Standardabweichungen'
                )
            outlier_children.append(dcc.Graph(figure=stl_z_fig))
            stl_fig = make_subplots(rows=4, cols=1)
            stl_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=stl_dec.trend,
                    mode='markers+lines',
                    name='Trend', 
                ), row=1, col=1
            )
            stl_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=stl_dec.seasonal,
                    mode='markers+lines',
                    name='Wiederkehrende Effekte',     
                ), row=2, col=1
            )
            stl_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=stl_dec.resid,
                    mode='markers+lines',
                    name='Andere Effekte',    
                ), row=3, col=1
            )
            stl_fig.add_trace(
                go.Scatter(
                    x=driver_timeframe['Date'],
                    y=driver_timeframe[outlier_choice],
                    mode='markers+lines',
                    name='Vollständiger Treiber',    
                ), row=4, col=1
            )
            stl_fig.update_layout(
                yaxis_tickformat = ',~r', yaxis2_tickformat = ',~r', 
                yaxis3_tickformat = ',~r', yaxis4_tickformat = ',~r',
                height=800, 
                title_text = 'Zerlegung in Trend, zyklische Effekte und Rest'
                )
            outlier_children.append(dcc.Graph(figure=stl_fig))
        outlier_children.append(
            dcc.Markdown(
                '''
                \* Hier zerlegen wir die Daten in Komponenten, die verschiedene Effekte  
                repräsentieren. Die Summe der drei Komponenten ergibt den gesamten Treiber.  
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
        )
        children[-1] = html.Div(children=outlier_children)
    
    if (comp_display and comp_choice is None) or (outlier_display and outlier_choice is None):
        children[-1] = html.Div() 
  
    if not comp_display and not outlier_display:
        graph = make_subplots(rows=len(choice), cols=1) 
        for i, c in enumerate(choice):
            graph.add_trace(
                go.Scatter(x = driver_timeframe['Date'], y = driver_timeframe[c], name = c), 
                row=i+1, col=1
            )
            graph.update_xaxes(title_text="Datum", row=i+1, col=1)
            graph.update_yaxes(title_text=c, row=i+1, col=1, tickformat = ',~r')
        graph.update_layout(height = 450*len(choice))
        children[-1] = dcc.Graph(figure=graph) 
   
    return children, store, comp_display or outlier_display, not comp_display, not outlier_display

@callback(
        Output('driver-selection', 'value'),
        Output('driver-dropdown', 'value'),
        Output('compare-selection', 'options'),
        Input('driverChoice', 'modified_timestamp'),
        State('driverChoice', 'data'),
        State('productChoice', 'data'), 
)
def on_driver(timeStamp, data_driver, data_product):
    if timeStamp is None:
        raise PreventUpdate
    
    if data_driver is None or not data_driver or data_driver.get('data') is None or len(data_driver.get('data'))==0:
        return [], [], []
    
    return data_driver.get('data'), data_driver.get('data'), data_driver.get('data')+[data_product.get('data')]
 
@callback(
    Output('WeiterTreiberauswahl', 'data'),
    Input('driver-selection', 'value'), 
    Input('driver-dropdown', 'value'),
    State('missing-message', 'children')
)
def toggle_button(choice, _, msg):
    result =  {
            'data': 'True'
    }
    if choice is None or not choice or len(choice)<2:
        return {}
    if msg:
        return {}
    return result

def calc_correlations(driver, choice, cutoff=0.6, return_correlation_matrix=False):
    cm = driver[choice].corr()
    corr_list = []
    for i in range(len(cm)):
        for j in range(i+1,len(cm)):
            corr_list.append([choice[i], choice[j], cm.iloc[i,j]])
    corr_df = pd.DataFrame(corr_list, columns=['Treiber A', 'Treiber B', 'Korrelation'])
    corr_df = corr_df.reindex(corr_df['Korrelation'].abs().sort_values(ascending=False).index)
    corr_df = corr_df[corr_df['Korrelation'].abs() >= cutoff].reset_index(drop=True)
    if return_correlation_matrix: 
        return corr_df, cm
    return corr_df
