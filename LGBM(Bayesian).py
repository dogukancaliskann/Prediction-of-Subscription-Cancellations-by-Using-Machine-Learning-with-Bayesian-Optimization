# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:41:13 2023

@author: dogukan1
"""

import pandas as pd
import numpy as np
from fast_ml.model_development import train_valid_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,StratifiedKFold
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier



def stratified_kfold_score(clf,X,y,n_fold):
    X,y = X.values,y.values
    strat_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    f1_list = []

    for train_index, test_index in strat_kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(x_train_fold, y_train_fold)
        preds = clf.predict(x_test_fold)
        f1_test = f1_score(preds,y_test_fold)
        f1_list.append(f1_test)

    return np.array(f1_list).mean()


def bo_params_lgbm(learning_rate,lambda_l1,lambda_l2,min_gain_to_split,max_depth,num_leaves,
                   bagging_fraction,n_estimators):
    
    params = {
        'learning_rate': learning_rate,
        'lambda_l1':lambda_l1,
        'lambda_l2':lambda_l2,
        'min_gain_to_split':min_gain_to_split,
        'max_depth':int(max_depth),
        'num_leaves':int(num_leaves),
        'bagging_fraction':bagging_fraction,
        'n_estimators':int(n_estimators)
        }
    
    clf = LGBMClassifier(learning_rate=params['learning_rate'],lambda_l1=params['lambda_l1'],lambda_l2=params['lambda_l2'],
                        min_gain_to_split=params['min_gain_to_split'],max_depth=params['max_depth'],num_leaves=params['num_leaves'],
                        bagging_fraction=params['bagging_fraction'],n_estimators=params['n_estimators'],bagging_freq=1)
   
    score = stratified_kfold_score(clf,X_train, y_train,5)
    return score






lgbm_bo = BayesianOptimization(bo_params_lgbm, {
                                              'learning_rate':(0.01,0.3),
                                              'lambda_l1':(1,20),
                                              'lambda_l2':(10,150),
                                              'min_gain_to_split':(1,5),
                                              'max_depth':(1,5),
                                              'num_leaves':(2,31),
                                              'bagging_fraction':(0.1,0.6),
                                              'n_estimators':(200,12000)
                                             })


results_lgbm = lgbm_bo.maximize(n_iter=200, init_points=20,acq='ei')


params_lgbm = lgbm_bo.max['params']
params_lgbm['n_estimators']= int(params_lgbm['n_estimators'])
print(params_lgbm)



lgbm_classifier = LGBMClassifier(bagging_fraction=0.3335181018927077,bagging_freq=1,lambda_l1=1.6999118585489774,lambda_l2= 50.871214793434845,
                                 learning_rate=0.22315403128129035,max_depth=4,min_gain_to_split=1.1706813074029352,n_estimators=6988,
                                 num_leaves=24)



lgbm_classifier.fit(X_train,y_train)

predictions_lgbm = lgbm_classifier.predict(X_test)

print(classification_report(y_test, predictions_lgbm))


















