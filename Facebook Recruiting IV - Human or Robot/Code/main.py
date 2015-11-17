__author__ = 'P_Kravik'

# Looks like pickly is really bad, this hdf5 thing might be better

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from facebook.data_cleaning import encode_predictions
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import copy


pd.set_option('display.width', 1000)

def facebook_modelling(derived_data_path):

    store = pd.HDFStore(derived_data_path + "data.h5")

    train = store['train']
    test = store['test']

    # set indices to merge
    train.set_index("bidder_id", inplace=True)
    test.set_index("bidder_id", inplace=True)

    store.close()

    store = pd.HDFStore(derived_data_path + "derived_data.h5")
    bidder = store['bidder_level'].join(pd.concat([train, test]), how="inner")
    store.close()


    # Merge everything. Only keeping inner (will need to later remerge on original files)
    # bidder = bidder_level.join(pd.concat([train, test]), how="inner")

    # indicator for whether in train or test dataset
    bidder['train'] = ~bidder['outcome'].isnull()
    bidder['test'] = ~bidder['train']

    for var in ['most_freq_country', 'most_freq_merchandise']:
        encoder = LabelEncoder().fit(bidder[var])
        bidder[var] = encoder.transform(bidder[var])

    # bidder = bidder[bidder.train == True]

    #To create:
    # num_bids/num_auctions
    bidder['bids_per_auction'] = bidder['num_bids'] / (bidder['num_auctions']+1)

    # num_bids/length_activity
    bidder['bids_per_time'] = bidder['num_bids'] / (bidder['length_activity']+1)

    # recode the dummies for tiny groups

    threshold = len(bidder)/100
    vars_to_recode = ['most_freq_device', 'most_freq_merchandise']
    for var in vars_to_recode:
        num = bidder[var].value_counts()
        small_val = num[num < threshold].index
        bidder.loc[bidder[var].apply(lambda x: x in small_val), var] = -1

    bidder.std_change_time.fillna(-1, inplace=True)
    bidder.mean_change_time.fillna(-1, inplace=True)

    # Model for each time period?
    # // 10 for 1st-3rd ip
    # in period 1
    # in period 2
    # in period 3
    # off-peak or peak times
    # change time
    # Dummy for more than one for all of the max

    continuous_vars = ['num_auctions',
                       'num_bids',
                       'num_url',
                       'num_country',
                       'num_device',
                       'num_merchandise',
                       'num_ip',
                       'num_first_bid',
                       'pct_first_bid',
                       'num_winner',
                       'pct_winner',
                       'num_outbid_self',
                       'num_outbid_self_diff_time',
                       'first_bid',
                       'last_bid',
                       'length_activity',
                       'num_periods',
                       'max_simul_actions',
                       'max_simul_auction',
                       'max_simul_country',
                       'max_simul_device',
                       'max_simul_ip',
                       'bids_per_auction',
                       'bids_per_time',
                       'most_common_auction_type',
                       'periods_seen',
                       'mean_change_time',
                       'std_change_time',
                       'std_pct_auction_bids',
                       'mean_pct_auction_bids',
                       'mean_num_bidders_auction',
                       'mean_num_bids_auction',
                       'mean_bids_timestamp',
                       'std_bids_timestamp']

    categorical_vars = ['most_freq_device',
                        'primary_period',
                        'most_freq_merchandise',
                        'most_freq_country']

    predictor_vars = continuous_vars + categorical_vars



    outcome_var = 'outcome'

    df = bidder[bidder.train == True]

    X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='binary', seed = 56, test_pct=0)

    test = [pd.get_dummies(df.pop(var)) for var in categorical_vars]

    run_random_forest_model(bidder[bidder.train == True], continuous_vars, categorical_vars, outcome_var)
    run_random_forest_model(bidder, predictor_vars, outcome_var, 999)

    generate_predictions(bidder, continuous_vars, categorical_vars, outcome_var, 'train', derived_data_path)
    encode_predictions(derived_data_path, 'that was unexpected binary gbm')


def convert_continuous_to_quantiles(df, varlist, n_quantiles=5):
    for var in varlist:
        df[var] = pd.qcut(df[var], n_quantiles, labels=False)


def merge_small_classes(df, varlist, num_top_classes):
    for var in varlist:
        top_classes = df.var.value_counts(num_top_classes)
        df.loc[df.var in top_classes, var] = -1


def test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='continuous', seed=56, test_pct=0.25):

    np.random.seed(seed)
    sampling = df.index.to_series().groupby(level=0).transform(lambda x: np.random.uniform(0, 1, 1))

    train_sample = sampling < 1 - test_pct

    y_test = df[~train_sample][outcome_var]
    y_train = df[train_sample][outcome_var]

    if categorical == 'binary':
        df_list = [df[continuous_vars]] + [pd.get_dummies(df[var], prefix=var) for var in categorical_vars]
        df = pd.concat(df_list, axis=1)
    elif categorical == 'continuous':
        df = df[continuous_vars + categorical_vars]
    else:
        raise KeyError('Wrong input')

    X_train = df[train_sample]
    X_test = df[~train_sample]

    return X_train, y_train, X_test, y_test


def run_random_forest_model(df, continuous_vars, categorical_vars, outcome_var, n_trees=1000, seed=42):

    X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='continuous', seed = 596, test_pct=0)

    rf = RandomForestClassifier(random_state=42)

    params = [{'max_depth': [8, 10, 12],
               'n_estimators': [1000],
               'min_samples_leaf': [5],
               'max_features': [3, 5, 10, 12],
               'class_weight': [None, 'auto']}]

    cv_strat = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=98)

    clf = GridSearchCV(rf, params, scoring='roc_auc', cv=cv_strat, n_jobs=3, verbose=1, iid=False)

    clf.fit(X_train, y_train)

    clf.grid_scores_
    clf.best_estimator_
    clf.best_params_
    clf.best_score_

    train_score = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    cv_score = cross_val_score(rf, X_train, y_train, scoring="roc_auc", cv=cv_strat).mean()
    # test_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("Train ROC: %f \n CV ROC: %f" % (train_score, cv_score))


def run_gradient_boosting_model(df, continuous_vars, categorical_vars, outcome_var):
    X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='binary', seed = 124, test_pct=.2)

    gbm = GradientBoostingClassifier(random_state=23)

    params = [{'learning_rate': [0.1, 0.01, 0.005],
               'n_estimators': [500],
               'max_depth': [3,  5, 8],
               'min_samples_leaf': [5],
               'max_features': [10, 15, 25, 35, 50],
               'subsample': [0.7, 1],
               'loss': ['deviance']}]

    cv_strat = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=98)

    clf = GridSearchCV(gbm, params, scoring='roc_auc', cv=cv_strat, n_jobs=3, verbose=1, iid=False)

    clf.fit(X_train, y_train)

    clf.grid_scores_
    clf.best_estimator_
    clf.best_params_
    clf.best_score_

    gbm = GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, min_samples_leaf=5, max_features=25, subsample=0.7)
    gbm.fit(X_train, y_train)
    roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])

    rf = RandomForestClassifier(n_estimators=1000, max_depth=10, max_features=15, min_samples_leaf=5, n_jobs=3, random_state=42)
    rf.fit(X_train, y_train)
    roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    test = FinalEstimator([gbm, rf], n_estimators=5)
    test.create_estimators(X_train, y_train, X_test)
    huh = test.create_prediction()

    X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='continuous', seed = 124, test_pct=.2)

    gbm = GradientBoostingClassifier(random_state=23)

    params = [{'learning_rate': [0.1, 0.01, 0.005],
               'n_estimators': [500],
               'max_depth': [3,  5, 8],
               'min_samples_leaf': [5],
               'max_features': [2, 5, 10, 15],
               'subsample': [0.7, 1],
               'loss': ['deviance']}]

    cv_strat = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=98)

    clf = GridSearchCV(gbm, params, scoring='roc_auc', cv=cv_strat, n_jobs=3, verbose=1, iid=False)

    clf.fit(X_train, y_train)

    clf.grid_scores_
    clf.best_estimator_
    clf.best_params_
    clf.best_score_

    gbm_continuous = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, max_features=15, min_samples_leaf=5)
    gbm_continuous.fit(X_train, y_train)
    roc_auc_score(y_test, gbm_continuous.predict_proba(X_test)[:, 1])

    rf_continuous = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=5, max_features=12, class_weight='auto')
    rf_continuous.fit(X_train, y_train)
    roc_auc_score(y_test, rf_continuous.predict_proba(X_test)[:, 1])

    tmp = FinalEstimator([gbm_continuous, rf_continuous], n_estimators=15)
    tmp.create_estimators(X_train, y_train, X_test)
    huh2 = tmp.create_prediction()
    roc_auc_score(y_test, huh2)

    return clf.best_score_, clf.best_params_


def run_continuous_categorical_models(df, continuous_vars, categorical_vars, outcome_vars, train_indicator):
    if train_indicator == "train":
        df_train, dfy_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='continuous', seed=56, test_pct=0)

        X_train = df_train[df[train_indicator] == True]
        y_train = dfy_train[df[train_indicator] == True]
        X_test = df_train[df[train_indicator] == False]
    else:
        X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='continuous', seed = 124, test_pct=0)

    gbm_continuous = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, max_features=15, min_samples_leaf=5)
    rf_continuous = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=5, max_features=12, class_weight='auto')
    final = FinalEstimator([gbm_continuous, rf_continuous], n_estimators=15)
    final.create_estimators(X_train, y_train, X_test)
    prediction = final.create_prediction()

    return prediction


def run_binary_categorical_models(df, continuous_vars, categorical_vars, outcome_vars, train_indicator):
    if train_indicator == "train":
        df_train, dfy_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='binary', seed=56, test_pct=0)

        X_train = df_train[df[train_indicator] == True]
        y_train = dfy_train[df[train_indicator] == True]
        X_test = df_train[df[train_indicator] == False]
    else:
        X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='binary', seed = 124, test_pct=0)



    gbm = GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, min_samples_leaf=5, max_features=25, subsample=0.7)
    rf = RandomForestClassifier(n_estimators=1000, max_depth=10, max_features=15, min_samples_leaf=5, n_jobs=3, random_state=42)
    final = FinalEstimator([gbm, rf], n_estimators=15)
    final.create_estimators(X_train, y_train, X_test)
    prediction = final.create_prediction()

    return prediction


def run_svm_prediction(df, continuous_vars, categorical_vars, outcome_vars, train_indicator):
    if train_indicator == "train":
        df_train, dfy_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='binary', seed=56, test_pct=0)

        X_train = df_train[df[train_indicator] == True]
        y_train = dfy_train[df[train_indicator] == True]
        X_test = df_train[df[train_indicator] == False]
    else:
        X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='binary', seed = 124, test_pct=0)

    svc = SVC(C=1, class_weight='auto', gamma=0, kernel='rbf', probability=True)
    scale = StandardScaler()
    svc.fit(scale.fit_transform(X_train), y_train)
    prediction = svc.predict_proba(scale.transform(X_test))[:, 1]

    return prediction


def run_final_models(df, continuous_vars, categorical_vars, outcome_var, train_indicator=""):
    continuous_prediction = run_continuous_categorical_models(df, continuous_vars, categorical_vars, outcome_var, train_indicator)
    binary_prediction = run_binary_categorical_models(df, continuous_vars, categorical_vars, outcome_var, train_indicator)
    svm_prediction = run_svm_prediction(df, continuous_vars, categorical_vars, outcome_var, train_indicator)

    predictions = np.mean([continuous_prediction, binary_prediction, svm_prediction], axis=0)

    predictions_df = pd.DataFrame({'bidder_id': df.index[df[train_indicator] == False],
                                   'prediction': predictions})

    predictions_df.to_csv(derived_data_path + 'predictions.csv', index=False)

    encode_predictions(derived_data_path, 'SoVeryClassy')


    return predictions


def generate_predictions(df, continuous_vars, categorical_vars, outcome_vars, train_indicator, derived_data_path):
    np.random.seed(46)
    df_train, dfy_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_vars, categorical='binary', seed=56, test_pct=0)

    X_train = df_train[df[train_indicator] == True]
    y_train = dfy_train[df[train_indicator] == True]
    X_test = df_train[df[train_indicator] == False]

    # rf = RandomForestClassifier(n_estimators=1000, max_depth=10, max_features=15, min_samples_leaf=5, n_jobs=3, random_state=42)
    # rf.fit(X_train, y_train)

    gbm = GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, min_samples_leaf=5, max_features=25, subsample=0.7)
    gbm.fit(X_train, y_train)

    gbm = BaggingClassifier(gbm, n_estimators=100, max_samples=0.95, random_state=77)
    gbm.fit(X_train, y_train)

    # rf = RandomForestClassifier(random_state=42, max_depth=8, max_features=100, min_samples_leaf=5, n_estimators=1000)
    # rf.fit(X_train, y_train)
    model = gbm

    X_test = df_train[df[train_indicator] == False]

    predictions = pd.DataFrame({'bidder_id': df.index[df[train_indicator] == False],
                               'prediction': model.predict_proba(X_test)[:, 1]})

    predictions.to_csv(derived_data_path + 'predictions.csv', index=False)


def feature_selection(df, predictor_var_full_list, outcome_var):
    num_features = len(predictor_var_full_list)
    cv_scores = [None] * num_features
    params_list = [None] * num_features

    baseline_cv_score, baseline_params = run_gradient_boosting_model(df, predictor_var_full_list, [], outcome_var)


    # For each feature, test removing it and calculate the cv score
    for i in range(0, num_features, 1):
        print "%d out of %d features" % (i+1, num_features)
        predictor_var = predictor_var_full_list[0:i] + predictor_var_full_list[i+1:]
        cv_score, params = run_gradient_boosting_model(df, predictor_var, [], outcome_var)
        cv_scores[i] = cv_score
        params_list[i] = params

    # If we wanted to go hardcore recursive, put this in the for loop
    if baseline_cv_score > max(cv_scores):
        best_score = baseline_cv_score
        best_params = baseline_params
        feature_list = predictor_var_full_list
        print "Best score is %f" % best_score
    else:
        max_index = cv_scores.index(max(cv_scores))
        feature_list = predictor_var_full_list[0:max_index] + predictor_var_full_list[max_index+1:]
        print "removed %s, best score: %f" % (predictor_var_full_list[max_index],  max(cv_scores))
        best_score, best_params, feature_list = feature_selection(df, feature_list, outcome_var)

    return best_score, best_params, feature_list


def run_svm_model(df, continuous_vars, categorical_vars, outcome_var):

    X_train, y_train, X_test, y_test = test_train_data(df, continuous_vars, categorical_vars, outcome_var, categorical='continuous', seed = 124, test_pct=0)

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)

    svc = SVC(probability=True, cache_size=500)

    params = [{'C': [10, 5, 1, 0.1, 0.001],
               'class_weight': ['auto'],
               'kernel': ['rbf'],
               'gamma': [0, 0.1, 0.01]}]

    cv_strat = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=98)

    clf = GridSearchCV(svc, params, scoring='roc_auc', cv=cv_strat, n_jobs=3, verbose=1, iid=False)

    clf.fit(X_train, y_train)

    clf.grid_scores_
    clf.best_estimator_
    clf.best_params_
    clf.best_score_

    roc_auc_score(y_test, clf.best_estimator_.predict_proba(scale.transform(X_test))[:,1])

    svc_predict = clf.best_estimator_.predict_proba(scale.transform(X_test))[:,1]

class FacebookPredictor(object):

    def __init__(self, model, n_estimators = 50):
        self.model = model
        self.n_estimators = n_estimators
        self.bagged_estimators = []

    def fit(self, X_train, y_train):
        for model in self.models:
            bag = BaggingClassifier(base_estimator=model, n_estimators=self.n_estimators, n_jobs=3, max_samples=0.95)
            bag.fit(X_train, y_train)
            self.bagged_estimators.append(bag)

    def predict(self, X_test):
        predictions = []
        for model in self.bagged_estimators:
            predictions.append(model.predict_proba(X_test)[:, 1])

        return np.mean(predictions, axis=0)

    pass

class FinalEstimator(object):

    def __init__(self, models, n_estimators = 50):
        self.models = models
        self.estimators = n_estimators
        self.param_grid = []
        self.predictions = []

    def create_estimators(self, X_train, y_train, X_test):
        for model in self.models:
            param_grid = self.create_parameter_grid(model)
            for parameters in param_grid:
                clf = BaggingClassifier(base_estimator=model.set_params(**parameters), n_estimators=self.estimators, max_samples=0.95, n_jobs = 3)
                clf.fit(X_train, y_train)
                prediction = clf.predict_proba(X_test)[:,1]
                self.predictions.append(prediction)

    def create_prediction(self):
        return np.mean(self.predictions, axis=0)


    def create_parameter_grid(self, model):
        params_grid = []
        optim_params = model.get_params()
        if isinstance(model, GradientBoostingClassifier):
            params_to_modify = ['learning_rate', 'n_estimators', 'min_samples_leaf', 'max_depth', 'max_features']
            for param in params_to_modify:
                if param == 'learning_rate':
                    high_param_value = optim_params[param] * 1.1
                    low_param_value = optim_params[param] * 0.9
                else:
                    high_param_value = int(max(round(optim_params[param] * 1.1), int(optim_params[param] + 1)))
                    low_param_value = int(min(round(optim_params[param] * 0.9), int(optim_params[param] - 1)))

                low_params = copy.copy(optim_params)
                low_params[param] = low_param_value

                params_grid.append(low_params)

                high_params = copy.copy(optim_params)
                high_params[param] = high_param_value

                params_grid.append(high_params)

        elif isinstance(model, RandomForestClassifier):
            params_to_modify = ['n_estimators', 'min_samples_leaf', 'max_depth', 'max_features']
            for param in params_to_modify:
                high_param_value = int(max(round(optim_params[param] * 1.1), optim_params[param] + 1))
                low_param_value = int(min(round(optim_params[param] * 0.9), optim_params[param] - 1))

                low_params = optim_params
                low_params[param] = low_param_value

                high_params = optim_params
                high_params[param] = high_param_value

                params_grid.append(low_params)
                params_grid.append(high_params)

        elif isinstance(model, SVC):


        return params_grid


if __name__ == '__main__':
    data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Facebook Recruiting IV - Human or Robot/Data/"
    # raw_data_path = data_path + "Raw/"
    derived_data_path = data_path + "Derived/"
    facebook_modelling(derived_data_path)

