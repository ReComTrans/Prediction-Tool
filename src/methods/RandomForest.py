import numpy as np
import pandas as pd
from itertools import chain
from xgboost import XGBRFRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
#from tscv import GapWalkForward
from methods.methods import methodname

def XGB_RandomForest(target_train,exo_train,exo_test,verbose=2):#, target_test):
     RF_feature_importance = pd.DataFrame()
     RF_pred = pd.DataFrame(index=exo_test.index)
     model = XGBRFRegressor(random_state=42)

     # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
     num_parallel_tree = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
     #
     # max_leaves = [2,5,8,10]
     min_child_weight = [1,3,7]
     # Maximum number of levels in tree
     max_depth = [int(x) for x in np.linspace(2, 10, num = 5)]
     # max_depth.append(None)
     gamma = [0,1,10]
     #
     learning_rate = [1]#[0.2, 0.5, 0.8]
     
     lambdas = [0, 0.001, 0.1, 1, 10, 100, 1e4, 1e7, 1e10]

     alphas = [0, 0.001, 0.1, 1, 10, 100, 1e4, 1e7, 1e10]

     # Method of selecting samples for training each tree
     random_grid = {'num_parallel_tree': num_parallel_tree,#'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    # 'max_leaves': max_leaves,
                    'min_child_weight': min_child_weight,
                    'learning_rate': learning_rate,
                    'verbosity': [verbose],
                    'subsample': [0.7],
                    'colsample_bynode': [0.7],
                    'gamma': gamma,
                    'lambda': lambdas, 
                    'alpha': alphas,
                    # 'num_boost_round': [1],
                    }

     model_RGS = RandomizedSearchCV(estimator = model, scoring="neg_mean_squared_error", param_distributions = random_grid, n_iter = 200, cv = 3, verbose=verbose, random_state=42, n_jobs = -1)

     RF_RGS_fit = model_RGS.fit(exo_train,target_train)#, eval_set=[(exo_train, target_train), (exo_test, target_test)])#,early_stopping_rounds=3)

     RF_RGS_best = RF_RGS_fit.best_estimator_
     print(f'bestes Model RGS: {RF_RGS_best}')
     Random_Params=RF_RGS_fit.best_params_
     print(f'Parameter von {methodname.XGB_RF} RGS: {Random_Params}')


     # n_estimators_grid = list(chain(range(Random_Params["n_estimators"]-200,Random_Params["n_estimators"],100), 
     #                     range(Random_Params["n_estimators"],Random_Params["n_estimators"]+300,100 )))
     step = int(Random_Params["num_parallel_tree"]*0.1)
     num_parallel_tree_grid = [Random_Params["num_parallel_tree"]-2*step, Random_Params["num_parallel_tree"]-step, 
                         Random_Params["num_parallel_tree"], Random_Params["num_parallel_tree"]+step, 
                         Random_Params["num_parallel_tree"]+2*step]

     
     if Random_Params["max_depth"] == None :
          max_depth_grid = [None]
     else:    
          max_depth_grid = [Random_Params["max_depth"]-1,Random_Params["max_depth"],Random_Params["max_depth"]+1]

     # if Random_Params['max_leaves'] == 2:
     #      max_leaves_grid = [1,2,3]
     # elif Random_Params['max_leaves'] == 5:
     #      max_leaves_grid = [4,5,6]
     # elif Random_Params['max_leaves'] == 8:
     #      max_leaves_grid = [7,8,9]
     # else:
     #      max_leaves_grid = [10,12,15]   

     if Random_Params['min_child_weight'] == 1.0:
          min_child_weight_grid = [0.5,1.0,1.5]
     elif Random_Params['min_child_weight'] == 3:
          min_child_weight_grid = [2,3,4]
     else:
          min_child_weight_grid = [5.0,7.0,10.0]
          
     if Random_Params['gamma'] == 0:
          gamma_grid = [0,0.1,0.5]
     elif Random_Params['gamma'] == 1:
          gamma_grid = [0.5,1,3]
     else:
          gamma_grid = [6,10,0.1*target_train.mean()[0]]


     # if Random_Params['learning_rate'] == 0.2:
     #      learning_rate_grid = [0.1,0.2,0.3]
     # elif Random_Params['learning_rate'] == 0.5:
     #      learning_rate_grid = [0.4,0.5,0.6]
     # else:
     #      learning_rate_grid = [0.7,0.8,0.9]
     learning_rate_grid = [1]

     grid = {'num_parallel_tree': num_parallel_tree_grid,#'n_estimators': n_estimators_grid,
             'max_depth': max_depth_grid,
          #    'max_leaves': max_leaves_grid,
             'min_child_weight': min_child_weight_grid,
             'learning_rate': learning_rate_grid,
             'verbosity': [verbose],
             'subsample': [0.7],
             'colsample_bynode': [0.7],
             'gamma': gamma_grid,
             'lambda': [Random_Params['lambda']], 
             'alpha': [Random_Params['alpha']],
          #    'num_boost_round': [1],
             }
    
     model_grid = GridSearchCV(estimator = model, scoring="neg_mean_squared_error", param_grid = grid, cv = 3, verbose=verbose,  n_jobs = -1)

     RF_CGS_fit = model_grid.fit(exo_train,target_train)#,early_stopping_rounds=3)
    
     RF_CGS_best = RF_CGS_fit.best_estimator_
     print(f'bestes Model CGS: {RF_CGS_best}')
     if model_grid.best_score_ >= model_RGS.best_score_:
          print(f'GridSearch wurde genutzt, Grid_score: {model_grid.best_score_} Random_score: {model_RGS.best_score_}')
          RF_pred[methodname.XGB_RF] = RF_CGS_fit.predict(exo_test)
          RF_feature_importance[methodname.XGB_RF] = pd.Series(RF_CGS_best.feature_importances_, index=exo_test.columns)
          RF_fitted_model = RF_CGS_fit
     else:
          print(f'RandomizedSearch wurde genutzt, Grid_score: {model_grid.best_score_} Random_score: {model_RGS.best_score_}')
          RF_pred[methodname.XGB_RF] = RF_RGS_fit.predict(exo_test)
          RF_feature_importance[methodname.XGB_RF] = pd.Series(RF_RGS_best.feature_importances_, index=exo_test.columns )
          RF_fitted_model = RF_RGS_fit
     RF_params = pd.DataFrame.from_dict(RF_fitted_model.best_params_, orient='index')
     RF_fit = pd.DataFrame(index=exo_train.index)
     RF_fit[methodname.XGB_RF] = RF_fitted_model.predict(exo_train)

     return RF_pred, RF_fit, RF_feature_importance, RF_params
