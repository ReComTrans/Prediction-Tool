import numpy as np
import pandas as pd
from itertools import chain
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from methods.methods import methodname

def XGBoost(target_train,exo_train,exo_test,verbose=1,cv=3):
    GB_feature_importance = pd.DataFrame()
    GB_pred = pd.DataFrame(index=exo_test.index)
    model = XGBRegressor(random_state=42)

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

    max_depth = [2,4,6]#[int(x) for x in np.linspace(1, 12, num = 6)]
    max_depth.append(None)

    learning_rate = [0.1, 0.2, 0.5]#, 0.8]

    subsample = [0.7]#[0.3, 0.5, 0.7, 0.9, 1]

    min_child_weight = [1,3,7]#[1.0,7.0]

    gamma = [0,1,10]#[0,0.15,0.5]

    #colsample_bytree = [0.3, 0.6, 0.9]

    #colsample_bylevel = [0.3, 0.6, 0.9]
    
    lambdas = [0, 0.001, 0.1, 1, 10, 100, 1e4, 1e7, 1e10]

    alphas = [0, 0.001, 0.1, 1, 10, 100, 1e4, 1e7, 1e10]

    param_grid = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        #'colsample_bytree': colsample_bytree,
        #'colsample_bylevel': colsample_bylevel,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'n_estimators': n_estimators,
        # 'base_score' : [0.5], 
        'booster': ['gbtree'],#,'dart','gblinear'], 
        # 'tree_method':['exact'],#,'auto','approx'],
        'num_parallel_tree': [1], 
        'verbosity': [verbose],
        'seed': [42], 
        'lambda': lambdas, 
        'alpha': alphas,
        }
    
    model_RGS = RandomizedSearchCV(estimator = model, scoring="neg_mean_squared_error", param_distributions = param_grid, n_iter = 200, cv = cv, refit=True, verbose=verbose, random_state=42, n_jobs = -1)

    GB_RGS_fit = model_RGS.fit(exo_train,target_train)#, eval_set=[(exo_train, target_train), (exo_test, target_test)])#,early_stopping_rounds=3)

    GB_RGS_best = GB_RGS_fit.best_estimator_
    print(f'bestes Model von {methodname.GB} RGS: {GB_RGS_best}')
    Random_Params = GB_RGS_fit.best_params_
    print(f'Parameter von {methodname.GB} RGS: {Random_Params}')

    # n_estimators_grid = list(chain(range(Random_Params["n_estimators"]-200,Random_Params["n_estimators"],100), 
    #                      range(Random_Params["n_estimators"],Random_Params["n_estimators"]+300,100 )))
    step = int(Random_Params["n_estimators"]*0.1)
    n_estimators_grid = [Random_Params["n_estimators"]-2*step, Random_Params["n_estimators"]-step, 
                         Random_Params["n_estimators"], Random_Params["n_estimators"]+step, 
                         Random_Params["n_estimators"]+2*step]
    
    if Random_Params["max_depth"] == None :
        max_depth_grid = [None]
    else:    
        max_depth_grid = [Random_Params["max_depth"]-1,Random_Params["max_depth"],Random_Params["max_depth"]+1]  
    
    if Random_Params['learning_rate'] == 0.2:
        learning_rate_grid = [0.1,0.2,0.3]
    elif Random_Params['learning_rate'] == 0.1:
        learning_rate_grid = [0.001,0.01,0.05,0.1]
    elif Random_Params['learning_rate'] == 0.5:
        learning_rate_grid = [0.4,0.5,0.6]
    else:
        learning_rate_grid = [0.7,0.8,0.9]

    #if Random_Params['subsample'] ==0.6:
    #    subsample_grid = [0.5,0.6,0.7]
    #else:
    #    subsample_grid = [0.8,0.9,1.0]
    subsample_grid = [Random_Params['subsample']]

    if Random_Params['min_child_weight'] == 1.0:
       min_child_weight_grid = [0.5,1.0,1.5]
    elif Random_Params['min_child_weight'] == 3:
        min_child_weight_grid = [2,3,4]
    else:
       min_child_weight_grid = [5.0,7.0,10.0]
    # min_child_weight_grid = min_child_weight

    #if Random_Params['gamma']==[0.15]:
    #    gamma_grid = [0.1,0.15,0.25]
    #elif Random_Params['gamma']==[0.5]:
    #    gamma_grid = [0.35, 0.5, 1.0]
    #else:
    #    gamma_grid = [0]
    if Random_Params['gamma'] == 0:
        gamma_grid = [0,0.1,0.5]
    elif Random_Params['gamma'] == 1:
        gamma_grid = [0.5,1,3]
    else:
        gamma_grid = [6,10,0.1*target_train.mean()[0]]
    # gamma_grid = gamma

    #if Random_Params['colsample_bytree'] == 0.3:
    #    colsample_bytree_grid = [0.2, 0.3, 0.4]
    #elif Random_Params['colsample_bytree'] == 0.6:
    #    colsample_bytree_grid = [0.5, 0.6, 0.7]
    #else:
    #    colsample_bytree_grid = [0.8, 0.9, 1.0]

    #if Random_Params['colsample_bylevel'] == 0.3:
    #    colsample_bylevel_grid = [0.2, 0.3, 0.4]
    #elif Random_Params['colsample_bylevel'] == 0.6:
    #    colsample_bylevel_grid = [0.5, 0.6, 0.7]
    #else:
    #    colsample_bylevel_grid = [0.8, 0.9, 1.0]

    grid = {'n_estimators': n_estimators_grid,
            'max_depth': max_depth_grid,
            'learning_rate': learning_rate_grid,
            'subsample': subsample_grid,
            'min_child_weight': min_child_weight_grid,
            'gamma': gamma_grid,
            #'colsample_bytree': colsample_bytree_grid,
            #'colsample_bylevel': colsample_bylevel_grid,
            'booster': ['gbtree'],#,'dart','gblinear'], 
            'tree_method': ['exact'],#,'auto','approx'],
            'num_parallel_tree': [1], 
            'verbosity': [verbose],
            'seed': [42], 
            'lambda': [Random_Params['lambda']],
            'alpha': [Random_Params['alpha']],
            }
    
    model_grid = GridSearchCV(estimator = model, scoring="neg_mean_squared_error", param_grid = grid, cv = cv, verbose=verbose, n_jobs = -1)

    GB_CGS_fit = model_grid.fit(exo_train,target_train.values.ravel())#,early_stopping_rounds=3)
    
    GB_CGS_best = GB_CGS_fit.best_estimator_
    print(f'bestes Model CGS: {GB_CGS_best}')
    if model_grid.best_score_ >= model_RGS.best_score_:
        print(f'GridSearch wurde genutzt, Grid_score: {model_grid.best_score_} Random_score: {model_RGS.best_score_}')
        GB_pred[methodname.GB] = GB_CGS_fit.predict(exo_test)
        GB_feature_importance[methodname.GB] = pd.Series(GB_CGS_best.feature_importances_, index=exo_test.columns)
        GB_fitted_model = GB_CGS_fit
    else:
        print(f'RandomizedSearch wurde genutzt, Grid_score: {model_grid.best_score_} Random_score: {model_RGS.best_score_}')
        GB_pred[methodname.GB] = GB_RGS_fit.predict(exo_test)
        GB_feature_importance[methodname.GB] = pd.Series(GB_RGS_best.feature_importances_, index=exo_test.columns )
        GB_fitted_model = GB_RGS_fit
    GB_params = pd.DataFrame.from_dict(GB_fitted_model.best_params_, orient='index')
    GB_fit = pd.DataFrame(index=exo_train.index)
    GB_fit[methodname.GB] = GB_fitted_model.predict(exo_train)

    return GB_pred, GB_fit, GB_feature_importance, GB_params
