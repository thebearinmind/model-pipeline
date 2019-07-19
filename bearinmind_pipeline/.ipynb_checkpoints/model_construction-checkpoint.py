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


class modelBuilder: 

    def __init__(self, problem_type, model_type):
        self.problem_type = problem_type
        self.model_type = model_type
    

    # Split target variable and remove all unnecessary columns
    def prepare_data_split(self, df_train, df_test, target, rem_cols, useVarImp = False, varimp_threshold = 100):
        if useVarImp:
            varImp = pd.read_csv(f'varImp/lgbm_importance.csv')
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
        return [X, Y, X_test]

    # show a pic of important variables and save the list to csv
    def show_varimp(self, fit_model, X):
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

    # Run the model: k fold cross validation, custom metric and different problems (calssification and regression). 
    # Currently only LightGBM and CatBoost are supported
    def run_model(self, X, Y, X_test, n_folds, metric_func = roc_auc_score, get_probab = False, save_varimp = False, params = None):
        kf = KFold(n_splits = n_folds, random_state = 1, shuffle = True)
        scores = []
        submit_pred = np.zeros(X_test.shape[0])
        for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
            # Create data for this fold
            Y_train, Y_valid = Y.iloc[train_index].copy(), Y.iloc[test_index].copy()
            X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()

            print( f'Fold: {i}')

            if self.problem_type == 'regression':

                if self.model_type == 'LGBM':
                    fit_model = lgb.LGBMRegressor(**params)
                    fit_model.fit(X_train, Y_train)

                elif self.model_type == 'CatBoost':
                    fit_model = cat.CatBoostRegressor(**params)                                                    
                    fit_model.fit(X_train, Y_train,  verbose = False)

                else : 
                     print(f'This {model_type} is not yet supported!')

            elif self.problem_type == 'classification':

                if self.model_type == 'LGBM':
                    fit_model = lgb.LGBMClassifier(**params)
                    fit_model.fit(X_train, Y_train)

                elif self.model_type == 'XGBoost':
                    fit_model = xgb.XGBClassifier(**params)
                    fit_model.fit(X_train, Y_train)

                else: 
                    print(f'{self.model_type} for classification is not yet supported')

            else: 
                print(f'{self.problem_type} type of a problem solving is not yet supported')

            if get_probab:
                pred = fit_model.predict_proba(X_valid)[:,1]
                submit_pred += fit_model.predict_proba(X_test)[:,1]/n_folds

            else:
                pred = fit_model.predict(X_valid)
                submit_pred += fit_model.predict(X_test)[:,1]/n_folds

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
        return [submit_pred, ave_score, fit_model]




