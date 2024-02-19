import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import dcc, html, callback, Output, Input, State

from methods.methods import methodname

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

#from dash_extensions.enrich import DashProxy, MultiplexerTransform, Output, Input, State #LogTransform <---- evtl noch interessant

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server
from a2wsgi import WSGIMiddleware
server = WSGIMiddleware(server)


sidebar = html.Div( 
            dbc.Nav( id = 'navbar',
                children= [
                    dbc.NavLink(
                        id="Uploader",
                        children=[
                            html.Div("Uploader", className="ms-2"),
                        ],
                        href='/',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    dbc.NavLink(
                        id="Voreinstellungen",
                        children=[
                            html.Div("Voreinstellungen", className="ms-2"),
                        ],
                        href='/Voreinstellungen',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    dbc.NavLink(
                        id="Produktauswahl",
                        children=[
                            html.Div("Produktauswahl", className="ms-2"),
                        ],
                        href='/Produktauswahl',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    dbc.NavLink(
                        id="Treiberauswahl",
                        children=[
                            html.Div("Treiberauswahl", className="ms-2"),
                        ],
                        href='/Treiberauswahl',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    dbc.NavLink(
                        id="Methodenauswahl",
                        children=[
                            html.Div("Methodenauswahl", className="ms-2"),
                        ],
                        href='/Methodenauswahl',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    dbc.NavLink(
                        id="Zusammenfassung",
                        children=[
                            html.Div("Zusammenfassung", className="ms-2"),
                        ],
                        href='/Zusammenfassung',
                        active="exact",
                        style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                        ),
                    ],
                
                vertical=True,
                pills=True,
            ),
            style= {"padding-right": "0.5rem", "border-right": "1px solid #777", "height": "80vh"}
        
        
)

app.layout = dbc.Container([
    dcc.Store(id = 'runNameStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'trainingDateStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'forecastDateStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'forecastingLabelWidthStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'granularityStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'seasonalityStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'unitsStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'productStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'productChoice', data = {}, storage_type = 'local'),
    dcc.Store(id = 'driverChoice', data = {}, storage_type = 'local'),
    dcc.Store(id = 'driverStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'variableTrainStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'exoTrainStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'exoTestStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'SzenarioStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'chosenMethodsStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Holt_Winters}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Holt_Winters}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.LASSO}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.LASSO}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.mlr}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.mlr}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.ridge}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.ridge}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.ElasticNet}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.ElasticNet}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Sarima}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Sarima}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Sarimax}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.Sarimax}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.XGB_RF}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.XGB_RF}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.GB}Store', data = {}, storage_type = 'local'),
    dcc.Store(id = f'{methodname.GB}ParaStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'predictionsStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'metricStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'summaryStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'resetBtnStore', data = {}, storage_type = 'local'),
    dcc.Store(id = 'WeiterUploader', data = {}, storage_type = 'local'), 
    dcc.Store(id = 'WeiterVoreinstellungen', data = {}, storage_type = 'local'),
    dcc.Store(id = 'WeiterProduktauswahl', data = {}, storage_type = 'local'),
    dcc.Store(id = 'WeiterTreiberauswahl', data = {}, storage_type = 'local'),


    dbc.Row([
        dbc.Col(html.Div("ReComTrans Prognosetool",
                         style={'fontSize':20, 'textAlign':'center'}))
    ]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2
                ),


            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10,
                ),
        ]
    )
], fluid=True)


@callback(
    Output('navbar', 'children'),
    Input('WeiterUploader', 'data'),
    Input('WeiterVoreinstellungen', 'data'),
    Input('WeiterProduktauswahl', 'data'),
    Input('WeiterTreiberauswahl', 'data'),


)
def render_pages(uploader, presets, product, driver):

    # children
    pages = []
    pages.append({
        'name': 'Uploader',
        'path': '/'
    })
    if(uploader != None and bool(uploader)):
        pages.append({
            'name': 'Voreinstellungen',
            'path': '/Voreinstellungen'
        })
    if(presets != None and bool(presets) ):
        pages.append({
            'name': 'Produktauswahl',
            'path': '/Produktauswahl'
        })
    if(product != None and bool(product) ):
        pages.append({
            'name': 'Treiberauswahl',
            'path': '/Treiberauswahl'
        })
    if(driver != None and bool(driver) ):
        pages.append({
            'name': 'Methodenauswahl',
            'path': '/Methodenauswahl'
        })
    pages.append({
        'name': 'Zusammenfassung',
        'path': '/Zusammenfassung'
        })
    children = []
    for page in pages:
        child =  dbc.NavLink(
                    id=page["name"],
                    children = [
                        html.Div(page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",
                    style={"margin-bottom": "0.5rem", "padding":"0.2rem"}
                    
                )
        children.append(child)
    return children
    
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3001)
