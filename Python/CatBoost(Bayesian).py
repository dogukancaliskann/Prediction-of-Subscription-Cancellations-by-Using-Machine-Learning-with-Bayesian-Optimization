# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:09:35 2023

@author: dogukan1
"""





categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
type(categorical_features_indices)
categorical_features_indices.dtype

df['ANNUAL_FEES'] = df['ANNUAL_FEES'].apply(np.int64)

def stratified_kfold_score3(clf,X,y,n_fold):
    X,y = X.values,y.values
    strat_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    f1_list = []

    for train_index, test_index in strat_kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(x_train_fold, y_train_fold,cat_features=categorical_features_indices)
        preds = clf.predict(x_test_fold)
        f1_test = f1_score(preds,y_test_fold)
        f1_list.append(f1_test)

    return np.array(f1_list).mean()


def bo_params_cat(learning_rate,depth,rsm,l2_leaf_reg,random_strength,max_leaves,bagging_temperature):
    
    params = {
        'learning_rate':learning_rate,
        'depth':int(depth),#1-5
        'rsm':rsm, #subsample 0.01-0.6
        'l2_leaf_reg':l2_leaf_reg, #1-4
        'random_strength':random_strength,#0.8-6.0
        'max_leaves':int(max_leaves),# #15-40  #growing_policiy=Lossguide olacak
        'bagging_temperature':bagging_temperature}#0.0-2.0
    
    clf = CatBoostClassifier(learning_rate=params['learning_rate'],depth=params['depth'],
                                 l2_leaf_reg=params['l2_leaf_reg'],random_strength=params['random_strength'],
                                 max_leaves=params['max_leaves'],bagging_temperature=params['bagging_temperature'],
                                 grow_policy="Lossguide")
   
    score = stratified_kfold_score3(clf,X_train, y_train,3)
    return score


cat_bo = BayesianOptimization(bo_params_cat, {
                                              'learning_rate':(0.01,0.2),
                                              'depth':(1,5),
                                              'rsm':(0.2,0.5),
                                              'l2_leaf_reg':(1,4),
                                              'random_strength':(0.8,6.0),
                                              'max_leaves':(15,40),
                                              'bagging_temperature':(0.0,6.0)
                                             })


results_cat = cat_bo.maximize(n_iter=10, init_points=2,acq='ei')


params_cat = cat_bo.max['params']
print(params_cat)


#catboost_classifier = CatBoostClassifier(bagging_temperature=1.9959195397956158,depth=4,l2_leaf_reg=2.6513370555622036,
                                         #random_strength=2.7272935456048626,learning_rate=0.1927071803841344,
                                         #max_leaves=27, rsm=0.21503768844984078,grow_policy="Lossguide")

catboost_classifier = CatBoostClassifier(bagging_temperature=3.0,depth=5,l2_leaf_reg=1.0,
                                         random_strength=6.0,learning_rate=0.2,
                                         max_leaves=18, rsm=0.2,grow_policy="Lossguide")

catboost_classifier.fit(X_train,y_train,cat_features=categorical_features_indices)

predictions_cat = catboost_classifier.predict(X_test)

print(classification_report(y_test, predictions_cat))





































