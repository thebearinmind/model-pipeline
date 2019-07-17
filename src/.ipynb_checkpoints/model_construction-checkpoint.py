import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import sklearn
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# Split target variable and remove all unnecessary columns
def prepare_data_split(df_train, df_test, target, rem_cols):
    rem_cols.append(target)
    X = df_train.drop(rem_cols, axis=1)
    Y = df_train[target]
    rem_cols.remove(target)
    X_test = df_test.drop(rem_cols, axis=1)
    return [X, Y, X_test]

# show a pic of important variables and save the list to csv
def show_varimp(fit_model, X):
        feature_imp = pd.DataFrame(sorted(zip(fit_model.feature_importances_,X.columns)), columns=['Value','Feature'])
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title(f'LightGBM Features of Model')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'varImp/lgbm_importance.png')
        varimps = pd.DataFrame(feature_imp.sort_values(by="Value", ascending=False))
        varimps.to_csv(f'varImp/lgbm_importance.csv', index = False)

# Run the model: k fold cross validation, custom metric and different problems (calssification and regression). 
# Currently only LightGBM and CatBoost are supported
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
            
            elif model_type == 'XGBoost':
                fit_model = xgb.XGBClassifier(**params)
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
    
    print(f'Displaying variable importance ...')
    show_varimp(fit_model, X)
        
    return [submit_pred, ave_score, fit_model]

def blend_results(df1, df2, _id, target, proport1 = 0.6):
    target1 = target + '_x'
    target2 = target + '_y'
    df1 = df1.merge(df2, on = 'id')
    
    dif = (df1[target1] - df1[target2]).abs().mean()
    print(f'the mean absolute difference between 2 submissions is: {dif}')
    
    df1[target] = (proport*df1[target1] + (1-proport)*df1[target2])
    
    df = df1[[_id, target]]
    
    return df

