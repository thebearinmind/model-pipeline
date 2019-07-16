import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import sklearn
import catboost as cat
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def encode_categorical(df_train, df_test, column_list):
    for f in column_list:
        df_train[f] = df_train[f].to_string()
        df_test[f] = df_test[f].to_string()        
        print(f'{f} column is being transformed ...')
        lbl = LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))

def prepare_data_split(df_train, df_test, target, rem_cols):
    rem_cols.append(target)
    X = df_train.drop(rem_cols, axis=1)
    Y = df_train[target]
    rem_cols.remove(target)
    X_test = df_test.drop(rem_cols, axis=1)
    return [X, Y, X_test]


def run_model(X, Y, X_test, n_folds, problem_type = 'regression', model_type = 'LGBM', metric_func = roc_auc_score, params = None):
    kf = KFold(n_splits = n_folds, random_state = 1, shuffle = True)
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # Create data for this fold
        Y_train, Y_valid = Y.iloc[train_index].copy(), Y.iloc[test_index].copy()
        X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()

        print( f'Fold: {i}')
        
        if problem_type == 'regression':
            
            if model_type == 'LGBM':
                fit_model = lgb.LGBMRegressor(**params)
                fit_model.fit(X_train, Y_train)

            elif model_type == 'CatBoost':
                fit_model = cat.CatBoostRegressor(**params)                                                    
                fit_model.fit(X_train, Y_train,  verbose = False)
                
            else : 
                 print(f'This {model_type} is not yet supported!')
        
        elif problem_type == 'classification':
            
            if model_type == 'LGBM':
                fit_model = lgb.LGBMClassifier(**params)
                fit_model.fit(X_train, Y_train)
            else: 
                print(f'{model_type} for classification is not yet supported')
        
        else: 
            print(f'{problem_type} type of a problem solving is not yet supported')

        pred = fit_model.predict(X_valid)
        
        # Save validation predictions for this fold
        score = metric_func(Y_valid, pred)
        
        print( f'The valuation metric for the fold {i} is {score}')
        scores.append(score)
    
    ave_score = np.mean(scores)

    print(f'The average score accross the folds is {ave_score}')
    
    submit_pred = fit_model.predict(X_test)
        
    return [submit_pred, ave_score]