import pandas as pd
import os

# Dieses file (Data.py) liegt in dataPreparation, folgender Pfad ist also src
Pfad = os.path.dirname(os.path.dirname(__file__))  
Ergebnisse_pfad = os.path.join(Pfad, 'results')
Dummy_data_pfad = os.path.join(os.path.join(os.path.dirname(Pfad), 'Daten'), 'Dummy')

Prediction_pfad = os.path.join(Ergebnisse_pfad, 'Prognosen')
Parameter_pfad = os.path.join(Ergebnisse_pfad, 'Parameter')

try:
    os.mkdir(Ergebnisse_pfad)
except:
    pass 
try:
    os.mkdir(Prediction_pfad)
except:
    pass 
try:
    os.mkdir(Parameter_pfad)
except:
    pass

def load_dummy_data():
    product = pd.read_excel(os.path.join(Dummy_data_pfad, 'Dummy-Daten-2004-2022-interpoliert.xlsx'))
    product.rename(columns={product.columns[0] :'Date'}, inplace=True)
    product['Date']=product['Date'].dt.strftime('%d-%m-%Y')
    product.index = product.Date.astype(str)
    product[product.columns[1:]] = product[product.columns[1:]].astype(float)
    return product

def load_dummy_drivers():
    driver = pd.read_excel(os.path.join(Dummy_data_pfad, 'Treiber-Dummy-Daten-interpoliert.xlsx'))
    driver.rename(columns={driver.columns[0] :'Date'}, inplace=True)
    driver['Date']=driver['Date'].dt.strftime('%d-%m-%Y')
    driver.index = driver.Date.astype(str)
    driver[driver.columns[1:]] = driver[driver.columns[1:]].astype(float)
    return driver


