# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:18:58 2023

@author: dogukan1
"""

def stratified_kfold_score1(clf,X,y,n_fold):
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



def bo_params_xgb(learning_rate,reg_lambda,reg_alpha,gamma,max_depth,
                  min_child_weight,subsample,colsample_bytree,scale_pos_weight,n_estimators):
    
    params = {
        'learning_rate': learning_rate,
        'reg_lambda':reg_lambda,
        'reg_alpha':reg_alpha,
        'gamma':gamma,
        'max_depth':int(max_depth),
        'min_child_weight':int(min_child_weight),
        'subsample':subsample,
        'colsample_bytree': colsample_bytree,
        'scale_pos_weight':scale_pos_weight,
        'n_estimators':int(n_estimators)
        }
    
    clf = XGBClassifier(learning_rate=params['learning_rate'],reg_lambda=params['reg_lambda'],reg_alpha=params['reg_alpha'],
                            gamma=params['gamma'],max_depth=params['max_depth'],min_child_weight=params['min_child_weight'],
                            subsample=params['subsample'],colsample_bytree=params['colsample_bytree'],scale_pos_weight=params['scale_pos_weight'],
                            n_estimators=params['n_estimators'],tree_method="gpu_hist",enable_categorical=True)
   
    score = stratified_kfold_score1(clf,X_train, y_train,5)
    return score

xgb_bo = BayesianOptimization(bo_params_xgb, {
                                              'learning_rate':(0.01,0.3),
                                              'reg_lambda':(10,150),
                                              'reg_alpha':(10,150),
                                              'gamma':(1,5),
                                              'max_depth':(1,5),
                                              'min_child_weight':(0,75),
                                              'subsample':(0.01,0.6),
                                              'colsample_bytree':(0.2,0.5),
                                              'scale_pos_weight':(2,2.50),
                                              'n_estimators':(90,150)
                                             })


results_xgb = xgb_bo.maximize(n_iter=200, init_points=20,acq='ei')


params_xgb = xgb_bo.max['params']
params_xgb['n_estimators']= int(params_xgb['n_estimators'])
print(params_xgb)




xgb_classifier = XGBClassifier(tree_method="gpu_hist", enable_categorical=True,colsample_bytree=0.5,gamma=1.0,
                               learning_rate=0.3,max_depth=5,min_child_weight=24.10030019230369,n_estimators=113,
                               reg_alpha=19.502704503985683,reg_lambda=111.93912050451571,scale_pos_weight=2.5,
                               subsample=0.6)

xgb_classifier.fit(X_train,y_train)

predictions_xgb = xgb_classifier.predict(X_test)

print(classification_report(y_test, predictions_xgb))











