# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:09:51 2023

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



df = pd.read_excel(r"C:\Users\dogukan1\Desktop\newdata22.xlsx")
factor = ['MEMBER_MARITAL_STATUS','MEMBER_GENDER','MEMBER_OCCUPATION_CD','MEMBERSHIP_PACKAGE','PAYMENT_MODE',
          'AGE_GROUP','TERM_GROUP','TARGET','START_MONTH']

for i in factor:
    df[i] = df[i].astype('category')


df.rename(
    columns={"START_MONTH": "START_SEASON"},
    inplace=True,
    )




X_train,X_test,y_train,y_test=train_test_split(df.drop(labels=['TARGET'], axis=1),
    df['TARGET'],
    test_size=0.1,
    random_state=0) 

rf = RandomForestClassifier()

rf.get_params()


#the function that returning the average F1-score of all samples, which is created with Stratified k-fold cross validation

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





#Optimization function for RF and its parameters
def bo_params_rf(max_samples,n_estimators,max_features,max_depth,min_samples_split):
    
    params = {
        'max_samples': max_samples,
        'max_features':max_features,
        'n_estimators':int(n_estimators),
        'max_depth':int(max_depth),
        'min_samples_split':int(min_samples_split)}
    
    clf = RandomForestClassifier(max_samples=params['max_samples'],max_features=params['max_features'],
                                 n_estimators=params['n_estimators'],max_depth=params['max_depth'],
                                 min_samples_split=params['min_samples_split'])
   
    score = stratified_kfold_score(clf,X_train, y_train,5)
    return score


#Optimizitation of RF


rf_bo = BayesianOptimization(bo_params_rf, {
                                              'max_samples':(0.1,1),
                                              'max_features':(0.5,1),
                                              'n_estimators':(80,300),
                                              'max_depth':(1,6),
                                              'min_samples_split':(2,10)
                                             })

#Results
results = rf_bo.maximize(n_iter=200, init_points=20,acq='ei')



# Parameters that maximizing Recall
params = rf_bo.max['params']
params['n_estimators']= int(params['n_estimators'])
print(params)



#Building new model

rf_v1 = RandomForestClassifier(max_samples=params['max_samples'],max_features=0.4,
                               n_estimators=params['n_estimators'],max_depth=6,
                               min_samples_split=2)




rf_v1.fit(X_train,y_train)

preds = rf_v1.predict(X_test)
print(classification_report(y_test,preds))















