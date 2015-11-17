__author__ = 'p_kravik'

import sys
sys.path.append('C:/Users/P_Kravik/Desktop/xgboost try 2/')
import xgboost as xgb



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import LeaveOneLabelOut

def get_out_of_fold_estimates(clf, X_train, y_train, cv):
    pred = pd.Series([0] * len(X_train))
    for train, test in cv:
        clf.fit(X_train.iloc[train], y_train.iloc[train])
        fold_pred = clf.predict(X_train.iloc[test])
        pred.iloc[test] = fold_pred

    return pred

def test_train_split_label(data, outcome, label, pct_test = 0.3, seed = 420):
    np.random.seed(seed)
    random_by_label = label.groupby(label).transform(lambda x: np.random.uniform())
    X_train = data[random_by_label >= pct_test].reset_index(drop=True)
    X_test = data[random_by_label < pct_test].reset_index(drop=True)
    y_train = outcome[random_by_label >= pct_test].reset_index(drop=True)
    y_test = outcome[random_by_label < pct_test].reset_index(drop=True)
    train_label = label[random_by_label >= pct_test]
    test_label = label[random_by_label < pct_test]

    return X_train, X_test, y_train, y_test, train_label, test_label

def test_train_cv_split_label(train_data, outcome, train_id, pct_test = 0.3, n_folds = 2, seed = 420):
    X_train, X_test, y_train, y_test, train_label, test_label = test_train_split_label(train_data, outcome, train_id, pct_test, seed)
    cv = cv_split_label(train_label, 'tube_assembly_id', n_folds)
    return X_train, X_test, y_train, y_test, cv


def cv_split_label(label, label_name, n_folds=2):
    np.random.seed(55)
    random_by_label = label.groupby(label).agg(lambda x: np.random.uniform())
    random_by_label.sort()

    num_ids = random_by_label.size
    cv_folds = pd.DataFrame({label_name: random_by_label.index,
                            'label': pd.Series([x * n_folds/(num_ids + 1) for x in range(1, num_ids + 1)])})

    merged = pd.DataFrame(label).merge(cv_folds, on=label_name, how='outer')

    cv_iterator = LeaveOneLabelOut(merged.label)

    return cv_iterator

train_data, test_data, outcome = get_train_data()
log_outcome = np.log(outcome + 1)
train_id = train_data.pop('tube_assembly_id')
test_id = test_data.pop('tube_assembly_id')

X_train, X_test, y_train, y_test, cv = test_train_cv_split_label(train_data, log_outcome, train_id, pct_test=0, n_folds=5, seed=151)

dtrain = xgb.DMatrix(np.array(train_data.values).astype(float), label=log_outcome.values)
dtest = xgb.DMatrix(np.array(test_data.values).astype(float))
num_rounds = 2000
param = {'objective': 'reg:linear',
         'learning_rate': 0.02,
         'min_child_weight': 2,
         'colsample_by_tree': 0.8,
         'max_delta_step': 2,
         'silent': 1,
         'subsample': 0.8,
         'max_depth': 8,
         'seed': 99}
models = [1] * 10
preds = [1] * 10
i = 0
for seed in [23, 45, 78, 299, 47, 67, 11, 87, 355, 64]:
    param['seed'] = seed
    plst = list(param.items())
    model = xgb.train(plst, dtrain, num_rounds)
    models[i] = model
    preds[i] = model.predict(dtest)
    i = i+1
    print i

#dtest = xgb.DMatrix(test_data.values.astype(float))

nypmat = np.array(X_train.values).astype(float)
dtrain = xgb.DMatrix(nypmat, label=y_train.values)
dtest = xgb.DMatrix(X_test.values.astype(float), label=y_test.values)
param = {'objective': 'reg:linear',
         'learning_rate': 0.02,
         'min_child_weight': 10,
         'colsample_by_tree': 0.8,
         'max_delta_step': 2,
         'silent': 1,
         'subsample': 0.8,
         'max_depth': 8,
         'seed': 420}
plst = list(param.items())
# param = {'objective': 'reg:linear',
#          'silent': 1,
#          'lambda': 100,
#          'alpha': 1}
from datetime import datetime
seeds = [11, 56, 88, 111, 52, 666, 900, 245, 145, 340]
pred = [1,2,3,4,5]
for i in range(0,5):
    param['seed'] = seeds[i]
    plst = list(param.items())

    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    #watchlist = [(dtrain, 'train')]

    num_rounds = 4000
    model = xgb.train(plst, dtrain, num_rounds, watchlist)

    pred[i] = model.predict(dtest)
    #print np.sqrt(mean_squared_error(y_test, pred[i]))
    print str(datetime.now())

watchlist = []

X_train = np.array(X_train).astype(float)
y_train = y_train.values
X_test = np.array(X_test).astype(float)
y_test = y_test.values

bag_models = [None] * 10
i=0
for model_seed in [11, 42, 66, 775, 323, 153, 463, 712, 675, 90]:
    gbm = xgb.XGBRegressor(max_depth=8,
                       learning_rate=0.02,
                       n_estimators=2000,
                       min_child_weight=6,
                       max_delta_step=2,
                       colsample_bytree=0.8,
                       subsample=0.8,
                       silent=1)
    gbm.set_params(seed = model_seed)
    bag_models[i] = gbm.fit(np.array(train_data).astype(float), np.power(np.expm1(log_outcome.values), 1.0/16))
    i=i+1

predictions = [None] * 10
i=0
for model in bag_models:
    predictions[i] = model.predict(np.array(test_data).astype(float))
    i=i+1

pred = np.mean(predictions, 0)

un_logged_pred = np.exp(pred) - 1

final = pd.DataFrame({'id': test_id.index + 1, 'cost': un_logged_pred})
final[['id', 'cost']].to_csv(folder + "../Predictions/power weirdness.csv", index=False)

def rmsle(true, pred):
    return np.sqrt(np.mean((np.log1p(true) - np.log1p(pred))**2))

# grid search

weights = X_train.quantity == 1
weights = [1 if x else 0.1 for x in (X_train.quantity == 1)]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import copy
def StackingEstimator(X_train, y_train, label):

    np.random.seed(55)
    random_by_label = label.groupby(label).agg(lambda x: np.random.uniform())
    random_by_label.sort()

    n_folds = 5
    label_name = 'tube_assembly_id'
    num_ids = random_by_label.size
    cv_folds = pd.DataFrame({label_name: random_by_label.index,
                            'label': pd.Series([x * n_folds/(num_ids + 1) for x in range(1, num_ids + 1)])})

    merged = pd.DataFrame(label).merge(cv_folds, on=label_name, how='outer')

    cv = LeaveOneLabelOut(merged.label)

    stacking_models = {
    'scikit_gbm': GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05, min_samples_split=6, max_depth=8, random_state=42),
    'scikit_rf': RandomForestRegressor(n_estimators=1000, max_depth=8, min_samples_leaf=5, max_features= 0.75, n_jobs=3, random_state=42),
    'xgboost_gbm': xgb.XGBRegressor(max_depth=8, learning_rate=0.02, n_estimators=2000, min_child_weight=6,
                                    max_delta_step=2, colsample_bytree=0.8, subsample=0.8, silent=1, seed=42),
    'xgboost_bigger_gbm': xgb.XGBRegressor(max_depth=8, learning_rate=0.02, n_estimators=4000, min_child_weight=6,
                                    max_delta_step=2, colsample_bytree=0.8, subsample=0.8, silent=1, seed=42),
    'xgboost_smaller_gbm': xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=6500, min_child_weight=6,
                                    max_delta_step=2, colsample_bytree=0.8, subsample=0.75, silent=1, seed=42)}

    # just gonna change the data for the second run
    # removed weight/length and weight/volume, also changed supplier to be number of obs, also no day of week

    data_type = 'datav2'

    multiple_outcomes = {'log': y_train,
                         'exp_16': np.power(outcome, 1.0/16)}# ,'normal': outcome, 'quantity_sum': outcome * X_train.quantity}

    multiple_outcomes = {'exp_18': np.power(outcome, 1.0 / 18),
                         'exp_20': np.power(outcome, 1.0 / 20)}

    multiple_outcomes = {'normal': outcome}

    stacking_models = {
    'xgboost_bigger_gbm_weight': xgb.XGBRegressor(max_depth=8, learning_rate=0.02, n_estimators=4000, min_child_weight=6,
                                    max_delta_step=2, colsample_bytree=0.8, subsample=0.8, silent=1, seed=42),
    'xgboost_smaller_gbm_wieght': xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=6500, min_child_weight=6,
                                    max_delta_step=2, colsample_bytree=0.8, subsample=0.75, silent=1, seed=42)}


    even_weights = 1.0 / X_train.num_tiers
    nonbracket_weights = [1 if x else 0 for x in (X_train.quantity == 1)]

    multiple_weights = {'even': even_weights,
                        'nonbracket': nonbracket_weights}

    nonbracket = True

    outsample_predictions = copy.deepcopy(merged)

    for model_name in stacking_models:
        for model_outcome in multiple_outcomes:
            run_name = model_name + '_' + model_outcome + '_' + data_type
            outsample_predictions[run_name] = 0

    for fold in range(0, n_folds):
        print "predicting on fold %d" % fold
        insample = outsample_predictions.label != fold
        outsample = outsample_predictions.label == fold
        for model_name, model in stacking_models.iteritems():
            print "fitting model..." + model_name
            for model_outcome in multiple_outcomes:

                print "outcome..." + model_outcome

                run_name = model_name + '_' + model_outcome + '_' + data_type
                model.fit(X_train[insample].values, multiple_outcomes[model_outcome][insample].values, sample_weights = even_weights[insample])
                pred = model.predict(X_train[outsample].values)
                if model_outcome == 'log':
                    scaled_pred = np.expm1(pred)
                elif model_outcome == 'exp_16':
                    scaled_pred = np.power(pred, 16)
                elif model_outcome == 'exp_18':
                    scaled_pred = np.power(pred, 18)
                elif model_outcome == 'exp_20':
                    scaled_pred = np.power(pred, 20)
                elif model_outcome == 'normal':
                    scaled_pred = pred
                else:
                    print "WARNING: unrecognized outcome..."
                scaled_pred.dump(folder + '../Stacking/first stage/individual folds/' + run_name + '_' + str(fold) + '.pkl')
                outsample_predictions.loc[outsample, run_name] = scaled_pred

        outsample_predictions.to_csv(folder + '../Stacking/first stage/individual folds/fold_' + str(fold) + '.csv')

    outsample_predictions.to_csv(folder + '../Stacking/first stage/first stage predictions new data weighted.csv')
    print "complete!"

    test_sample_predictions = {'id': test_id}
    for model_name, model in stacking_models.iteritems():
        for model_outcome in multiple_outcomes:

            run_name = model_name + '_' + model_outcome + '_' + data_type

            model.fit(X_train.values, multiple_outcomes[model_outcome], sample_weights = even_weights)
            test_pred = model.predict(X_test.values)
            if model_outcome == 'log':
                scaled_pred = np.expm1(test_pred)
            elif model_outcome == 'exp_16':
                scaled_pred = np.power(test_pred, 16)
            elif model_outcome == 'exp_18':
                scaled_pred = np.power(test_pred, 18)
            elif model_outcome == 'exp_20':
                scaled_pred = np.power(test_pred, 20)
            elif model_outcome == 'normal':
                scaled_pred = test_pred
            scaled_pred.dump(folder + '../Stacking/first stage/test data predictions/' + run_name + '.pkl')
            test_sample_predictions[run_name] = scaled_pred

    test_predictions = pd.DataFrame(test_sample_predictions)
    test_predictions.to_csv(folder + '../Stacking/first stage/test sample predictions new data and weights.csv')

    for col in outsample_predictions.columns:
        if col not in ['tube_assembly_id', 'label']:
            print col + "cv error: %f" % rmsle(outcome, outsample_predictions[col])


    comb = np.mean(outsample_predictions[[col for col in outsample_predictions.columns if col not in ['tube_assembly_id', 'label', 'scikit_rf_exp_16', 'scikit_rf_log', 'scikit_gbm_exp_16cv', 'scikit_gbm_log']]], 1)
    print "combined cv error: %f" % rmsle(outcome, comb)

    outsample_predictions = pd.read_csv(folder + '../Stacking/first stage/combined first stage predictions.csv', )


    important_meta_features = ['quantity', 'supplier', 'num_quantity_tiers', 'total_volume', 'annual_usage_per_quantity', 'annual_usage_per_min_quantity', 'quote_date_year',
                               'quote_date_week', 'length_per_bend_radius', 'annual_usage', 'length_per_bend', 'combined_weight']
    X_train_second_stage = copy.deepcopy(outsample_predictions[[col for col in outsample_predictions.columns if col not in ['tube_assembly_id', 'label']]])
    X_train_second_stage = copy.deepcopy(outsample_predictions[[col for col in outsample_predictions.columns if col not in ['tube_assembly_id', 'label'] and 'log' in col and 'scikit' not in col]])

    X_train_second_stage = np.log1p(X_train_second_stage)
    X_train_second_stage = pd.concat([X_train_second_stage, X_train[['quantity', 'supplier', 'num_tiers']]], axis=1)

    for col in ['xgboost_smaller_gbm_wieght_normal_datav2', 'xgboost_bigger_gbm_weight_normal_datav2']:
        X_train_second_stage.loc[X_train_second_stage[col] < 0, col] = X_train_second_stage.loc[X_train_second_stage[col] < 0, 'xgboost_bigger_gbm_exp_20_datav2']

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.grid_search import GridSearchCV

    second_stage = KNeighborsRegressor(n_neighbors=100, weights='distance', leaf_size=30)


    second_stage = ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import make_scorer

    scl = StandardScaler()

    second_stage = Ridge(alpha=50)
    parameters = {'alpha':[1000, 100, 50, 25, 10, 5, 1, 0.1, 0.01, 0.001]}
    clf = GridSearchCV(second_stage, parameters, scoring='mean_squared_error')
    clf.fit(scl.fit_transform(X_train_second_stage), y_train)
    clf.grid_scores_


    second_stage = xgb.XGBRegressor(n_estimators = 200, max_depth = 4, learning_rate=0.04 ,silent=True, min_child_weight = 6, colsample_bytree=0.1, subsample=0.8, max_delta_step = 2, seed=420)
    second_stage = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, max_features=0.2, n_jobs=3, random_state=42)
    parameters = {'max_depth': [4, 5, 6, 7],
                  'max_features': [0.1, 0.2, 0.3, 0.4]}
    clf = GridSearchCV(second_stage, parameters, scoring='mean_squared_error')
    clf.fit(X_train_second_stage, y_train)
    clf.grid_scores_


    for col in X_train_second_stage.columns:
        if col not in ['tube_assembly_id', 'label']:
            print col + "cv error: %f" % np.sqrt(mean_squared_error(y_train, X_train_second_stage[col]))

    second_stage.fit(X=X_train_second_stage, y=y_train)
    np.sqrt(mean_squared_error(y_train, second_stage.predict(X_train_second_stage)))

    cv_results = np.sqrt(-1 * cross_val_score(bag, X_train_second_stage, y_train, cv=cv, scoring='mean_squared_error'))
    print np.mean(cv_results)

    bag = BaggingRegressor(base_estimator=second_stage, n_estimators=200, random_state=42, max_samples=0.9)

    second_stage_prediction = np.expm1(second_stage.predict(test_predictions[['xgboost_bigger_gbm', 'xgboost_smaller_gbm', 'xgboost_gbm']]))

    final = pd.DataFrame({'id': test_id.index + 1, 'cost': second_stage_prediction})
    final[['id', 'cost']].to_csv(folder + "../Predictions/stacking test.csv", index=False)

def read_test_predictions():
    stacking_folder = folder + '../Stacking/first stage/'
    exp_pred = np.power(pd.read_csv(stacking_folder + 'test sample predictions exp.csv').drop('id', 1), 16)
    log_pred = np.expm1(pd.read_csv(stacking_folder + 'test sample predictions log.csv').drop('id', 1))
    weighted_pred = pd.read_csv(stacking_folder + 'test sample predictions new data and weights.csv').drop('id', 1)
    new_data_pred = pd.read_csv(stacking_folder + 'test sample predictions new data.csv')

    test_pred = pd.concat([exp_pred, log_pred, weighted_pred, new_data_pred], 1)

    for col in ['xgboost_smaller_gbm_wieght_normal_datav2', 'xgboost_bigger_gbm_weight_normal_datav2']:
        test_pred.loc[test_pred[col] < 0, col] = test_pred.loc[test_pred[col] < 0, 'xgboost_bigger_gbm_exp_20_datav2']

    test_prediction = second_stage.predict(test_pred[X_train_second_stage.columns])

    final = pd.DataFrame({'id': test_pred.index + 1, 'cost': test_prediction})
    final[['id', 'cost']].to_csv(folder + "../Predictions/ridge stacking test.csv", index=False)

