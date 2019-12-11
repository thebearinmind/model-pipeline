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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class modelBuilder: 

    def __init__(self, problem_type, model_type):
        self.problem_type = problem_type
        self.model_type = model_type
    

    
    def prepare_data_split(self, df_train, df_test, target, rem_cols, useVarImp = False, varimp_threshold = 100):
        """
        Split target variable and remove all unnecessary columns
        
        in case useVarImp is set to True, features will be filered out by the list of the variable importances and 
        the specified threshold. This can olny be done if there was one round of iteration of the model and 
        variable importances were saved in varImp directory.        
        
        """
        
        if useVarImp:
            varImp = pd.read_csv(f'varImp/{self.model_type}_importance.csv')
            varImp = varImp[varImp.Value >= varimp_threshold]
            varImpCount = varImp["Feature"].nunique() 
            varImpUnique = list(varImp['Feature'].unique())
            varImpUnique.extend(rem_cols)
            varImpUnique.extend([target])

            df_train = df_train[varImpUnique]
            varImpUnique.remove(target)

            df_test = df_test[varImpUnique]

            print(f'{varImpCount} features have been chosen for modeling')

        else : 
            pass
        rem_cols.append(target)
        X = df_train.drop(rem_cols, axis=1)
        Y = df_train[target]
        rem_cols.remove(target)
        X_test = df_test.drop(rem_cols, axis=1)
        
        print(f'{X.shape[1]} features have been chosen for modeling')
        
        prep_data = {'X': X,
                     'Y': Y,
                     'X_test':X_test}
        
        return prep_data

    
    def show_varimp(self, fit_model, X):
            """
            Save the list of the variable importance to the varImp directory. Additionally, the plot with the imporances will be
            displayed in the console.            
            
            """
            
            if not os.path.exists('varImp'):
                os.makedirs('varImp')

            feature_imp = pd.DataFrame(sorted(zip(fit_model.feature_importances_,X.columns)), columns=['Value','Feature'])
            plt.figure(figsize=(20, 10))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title(f'LightGBM Features of Model')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'varImp/lgbm_importance.png')
            varimps = pd.DataFrame(feature_imp.sort_values(by="Value", ascending=False))
            varimps.to_csv(f'varImp/lgbm_importance.csv', index = False)
    
    def hyperparam_tuning(self, prep_data, search_type, param_grid, n_iter_hp, metric, n_folds, params = None):
        """
        Function to enable hyperparameters tunning. 
        
        Allowed type of searches: randomized, grid.
        
        """
        X = prep_data['X']
        Y = prep_data['Y']
        X_test = prep_data['X_test']
        
        if self.problem_type == 'regression':
            
            if self.model_type == 'LGBM':
                fit_model = lgb.LGBMRegressor(**params)
            
            elif self.model_type == 'CatBoost':
                fit_model = cat.CatBoostRegressor(**params)                                                    
            
            elif self.model_type == 'XGBoost':
                fit_model = xgb.XGBRegressor(**params)

            else : 
                print(f'This {self.model_type} is not yet supported!')

        elif self.problem_type == 'classification':

            if self.model_type == 'LGBM':
                fit_model = lgb.LGBMClassifier(**params, verbose = verbose)

            elif self.model_type == 'XGBoost':
                fit_model = xgb.XGBClassifier(**params)

            else: 
                print(f'{self.model_type} for classification is not yet supported')

        else: 
            print(f'{self.problem_type} type of a problem solving is not yet supported')
        
        if search_type == 'randomized':
        
            gs = RandomizedSearchCV(estimator=fit_model, param_distributions=param_grid, n_iter=n_iter_hp, scoring=metric, cv=n_folds, refit=True, random_state=314, verbose=True)
        
        elif search_type == 'grid':
            
            gs = GridSearchCV(estimator=fit_model, param_distributions=param_grid, n_iter=n_iter_hp, scoring=metric, cv=n_folds, refit=True, random_state=314, verbose=True)
            
        else:
            print(f'{self.search_type} type of a search is not yet supported')
        
        gs.fit(X, Y)

        print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

        return(gs)

    def run_model(self, prep_data, n_folds, metric_func = roc_auc_score, get_probab = False, save_varimp = False, params = None, oversmp = False, verbose = 2, 
                  cv_shuffle = True, hyperparam_tunning = False, search_type = None, param_grid = None, n_iter_hp = None, metric = None):
        """
        The model based on the problem type (regression or classification) as well as chosen algorithm (e.g. LightGBM) will be run across specified
        number of K folds. The metric will be displayed and saved in a list for each single fold. The final score with be represented as an average 
        over the folds.
        
        In case of classification problem, the flag get_probab = True will return probabilities for each class.
        
        In addition, if oversample is set to true, the positive class will be oversampled based on SMOTE algorithm.
        
        save_varimp pararmeter set to True, saves variable importance of the model to the varImp directory.
        """
        
        X = prep_data['X']
        Y = prep_data['Y']
        X_test = prep_data['X_test']
        
        kf = KFold(n_splits = n_folds, random_state = 1, shuffle = cv_shuffle)
        scores = []
        submit_pred = np.zeros((X_test.shape[0],1))
        
        if hyperparam_tunning:
            gs = self.hyperparam_tuning(prep_data = prep_data, search_type = search_type, param_grid = param_grid, n_iter_hp = n_iter_hp, metric = metric, n_folds = n_folds, params = params)
        else:
            pass
        
        for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
            # Create data for this fold
            Y_train, Y_valid = Y.iloc[train_index].copy(), Y.iloc[test_index].copy()
            X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
            
            if oversmp:
                sm = SMOTE(random_state=12, ratio = 1.0)
                x_train_res, y_train_res = sm.fit_sample(X_train,  Y_train)
                X_train = pd.DataFrame(x_train_res)
                Y_train = pd.Series(y_train_res)
            else:
                pass
                
            print( f'Fold: {i}')

            if self.problem_type == 'regression':

                if self.model_type == 'LGBM':
                    fit_model = lgb.LGBMRegressor(**params)
                    
                elif self.model_type == 'CatBoost':
                    fit_model = cat.CatBoostRegressor(**params)                                                    
                
                elif self.model_type == 'XGBoost':
                    fit_model = xgb.XGBRegressor(**params)

                else : 
                     print(f'This {self.model_type} is not yet supported!')

            elif self.problem_type == 'classification':

                if self.model_type == 'LGBM':
                    fit_model = lgb.LGBMClassifier(**params, verbose = verbose)

                elif self.model_type == 'XGBoost':
                    fit_model = xgb.XGBClassifier(**params)

                else: 
                    print(f'{self.model_type} for classification is not yet supported')

            else: 
                print(f'{self.problem_type} type of a problem solving is not yet supported')

            if hyperparam_tunning:
                fit_model = gs.best_estimator_
                fit_model.fit(X_train, Y_train)
            else:
                fit_model.fit(X_train, Y_train)
            
            if get_probab:
                pred = fit_model.predict_proba(X_valid)[:,1]
                submit_pred += fit_model.predict_proba(X_test)[:,1]/n_folds

            else:
                
                full_fold_pred = fit_model.predict(X_test)
                full_fold_pred = np.reshape(full_fold_pred, (full_fold_pred.shape[0],1))
                pred = fit_model.predict(X_valid)
                submit_pred += full_fold_pred/n_folds
                
            
            # Save validation predictions for this fold
            score = metric_func(Y_valid, pred)

            print( f'The valuation metric for the fold {i} is {score}')
            scores.append(score)

        ave_score = np.mean(scores)

        print(f'The average score accross the folds is {ave_score}')

        if save_varimp:
            print(f'Displaying variable importance ...')
            self.show_varimp(fit_model, X)
        
        else:
            pass        
        
        model_result = {
            'FinalPrediction': submit_pred,
            'AverageScore' : ave_score,
            'FinalModel' : fit_model      
        }
        
        
        return model_result




