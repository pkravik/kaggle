import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cPickle as pickle


def data_prep(derived_data_path):
    """
    Runs all of the data prep functions
    :param derived_data_path: path to folder to save hdf
    :return:
    """
    raw_store = pd.HDFStore(derived_data_path + "data.h5")
    new_store = pd.HDFStore(derived_data_path + "derived_data.h5")

    # should eventually loop through?
    print("Creating bids variables")
    create_and_save(create_bids_variables, raw_store, new_store, 'bids', 'newbids')

    print("Creating simultaneous variables")
    create_and_save(create_simultaneous_actions_df, raw_store, new_store, 'bids', 'simultaneous_df')

    store.close()


def clean_data(raw_data_path, derived_data_path):
    print("Loading data...")
    train = pd.read_csv(raw_data_path + "train.csv")
    test = pd.read_csv(raw_data_path + "test.csv")
    bids = pd.read_csv(raw_data_path + "bids.csv")

    print("Encoding IDs...")
    id_encoder = LabelEncoder().fit(train["bidder_id"].append(test["bidder_id"]).append(bids["bidder_id"]))

    for df in [train, test, bids]:
        df["bidder_id"] = id_encoder.transform(df["bidder_id"])

    print("Encoding account vars...")
    account_encoders = {}
    for var in ["payment_account", "address"]:
        encoder = LabelEncoder().fit(train[var].append(test[var]))
        train[var] = encoder.transform(train[var])
        test[var] = encoder.transform(test[var])
        account_encoders[var] = encoder

    print("Encoding bid vars...")
    bids_encoders = {}

    for var in ["auction", "device", "url"]:
        encoder = LabelEncoder().fit(bids[var])
        bids_encoders[var] = encoder
        bids[var] = encoder.transform(bids[var])

    all_encoders = {'id': id_encoder,
                    'account': account_encoders,
                    'bids': bids_encoders}

    print("Saving encoders and data...")
    pickle.dump(all_encoders, open(derived_data_path + "Encoders/all_encoders.p", "wb"))
    pickle.dump(bids, open(derived_data_path + "bids.p", "wb"))
    pickle.dump(train, open(derived_data_path + "train.p", "wb"))
    pickle.dump(test, open(derived_data_path + "test.p", "wb"))


def save_to_hdf(derived_data_path):
    clean_data(data_path + "Raw/", data_path + "Derived/")

    bids = pickle.load(open(derived_data_path + "bids.p", "rb"))
    train = pickle.load(open(derived_data_path + "train.p", "rb"))
    test = pickle.load(open(derived_data_path + "test.p", "rb"))
    # encoder = pickle.load(open(derived_data_path + "Encoders/all_encoders.p", "rb"))

    store = pd.HDFStore(derived_data_path + "data.h5")
    store['bids'] = bids
    store['train'] = train
    store['test'] = test
    store.close()


def data_prep(derived_data_path):
    """
    Runs all of the data prep functions
    :param derived_data_path: path to folder to save hdf
    :return:
    """
    raw_store = pd.HDFStore(derived_data_path + "data.h5")
    new_store = pd.HDFStore(derived_data_path + "derived_data.h5")

    # should eventually loop through?
    print("Creating bids variables")
    create_and_save(create_bids_variables, raw_store, new_store, 'bids', 'newbids')

    print("Creating simultaneous variables")
    create_and_save(create_simultaneous_actions_df, raw_store, new_store, 'bids', 'simultaneous_df')

    raw_store.close()
    new_store.close()


def create_and_save(fn, raw_store, new_store, df_name, new_df_name):
    """
    Call an arbitrary dataframe creation function and saves it in an HDFStore
    :param fn: function that takes a single df as an argument and returns a new dataframe
    :param store: HDFstore to read and write results
    :param df_name: name of dataframe in HDFstore to pass to fn
    :param new_df_name: name of new dataframe stored in th HDFStore
    :return:
    """
    # store[new_df_name] = fn(store[df_name])
    new_store.put(new_df_name, fn(raw_store[df_name]))


# ACCOUNT LEVEL

def classify_auction_type(x):
    min_time = x['min'] / (1.0 * 10**15)
    max_time = x['max'] / (1.0 * 10**15)
    if 9.62 < min_time < 9.66 and 9.66 > max_time > 9.62:
        return 1
    elif 9.68 < min_time < 9.72 and 9.68 < max_time < 9.72:
        return 2
    elif 9.62 < min_time < 9.66 and 9.68 < max_time < 9.72:
        return 3
    elif 9.74 < min_time < 9.78 and 9.74 < max_time < 9.78:
        return 4
    else:
        raise ValueError('Value out of bounds')


def create_bids_variables(bids):
    """
    Creates new columns that are later used for
    :param bids: pass in the old bids dataframe
    :return: a new dataframe with more columns
    """
    # bids.time = bids.time - min(bids.time)

    # Sort by auction and bid time
    bids.sort(["auction", "time"], ascending=True, inplace=True)

    # Create indicator for the first bid of an auction
    lagged_auction = bids["auction"].diff()
    bids["first_bid"] = lagged_auction != 0
    bids["last_bid"] = bids.groupby("auction")['time'].transform(lambda x: (x == max(x)))

    period = bids.groupby('auction').time.agg(['min', 'max']).apply(classify_auction_type, axis=1).reset_index()
    period.columns = ['auction', 'period']
    bids = bids.merge(period, how='left', on='auction')

    periods_seen = bids.groupby('bidder_id').period.apply(lambda x: str(sorted(x.unique())))
    encoded_periods_seen = pd.DataFrame({'bidder_id': periods_seen.index, 'periods_seen': LabelEncoder().fit_transform(periods_seen)})

    bids = bids.merge(encoded_periods_seen, on='bidder_id', how='left')


    # Create a column with the time since the last bid for each bid
    bids["change_time"] = bids["time"].diff()
    bids.loc[bids.first_bid == True, "change_time"] = np.NaN
    bids.loc[bids.change_time > 0.2 * 10**13, "change_time"] = np.NaN

    # Whether they outbid themselves
    bids["same_bidder"] = bids["bidder_id"].diff() == 0
    bids.loc[bids.first_bid == True, "same_bidder"] = False

    bids["same_bidder_diff_time"] = (bids["bidder_id"].diff() == 0) & (bids['change_time'] != 0)

    return bids


def create_simultaneous_actions_df(bids):
    """
    This creates a dataframe with the number of simultaneous actions and unique devices/country/ip/auctions
    :param bids: raw bids dataframe
    :return: New dataframe with same rows and 5 columns, indexed by bid if
    """
    # Get the number of simultaneous bids by bidder and timestamp
    simultaneous_actions = bids.groupby(["bidder_id", "time"])['time'].transform('size')

    # Get number unique auction, device, ip, and country. Only doing for those with multiple actions to speed up and
    # re-merge later
    simultaneous_bids = bids[simultaneous_actions > 1].groupby(["bidder_id", "time"])

    simultaneous_auction = simultaneous_bids['auction'].transform(lambda x: x.nunique())
    simultaneous_device = simultaneous_bids['device'].transform(lambda x: x.nunique())
    simultaneous_ip = simultaneous_bids['ip'].transform(lambda x: x.nunique())
    simultaneous_country = simultaneous_bids['country'].transform(lambda x: x.nunique())

    # Combine all of the results into a dataframe that can later be merged onto bids.
    simultaneous_df = pd.DataFrame({'simul_actions': simultaneous_actions,
                                    'simul_auction': simultaneous_auction,
                                    'simul_device': simultaneous_device,
                                    'simul_ip': simultaneous_ip,
                                    'simul_country': simultaneous_country})

    return simultaneous_df


def breakdown_ip_address(ip_addresses):
    # this function takes a pandas series of ip addresses and returns a dataframe with the four numbers

    first_ip = ip_addresses.str.extract('^(\d+).')
    second_ip = ip_addresses.str.extract('^\d+.(\d+).')
    third_ip = ip_addresses.str.extract('^\d+.\d+.(\d+).')
    fourth_ip = ip_addresses.str.extract('^\d+.\d+.\d+.(\d+).')

    ip_df = pd.DataFrame({'first_ip': first_ip,
                          'second_ip': second_ip,
                          'third_ip': third_ip,
                          'fourth_ip': fourth_ip})

    return ip_df


def encode_predictions(derived_data_path, name):
    store = pd.HDFStore(derived_data_path + "data.h5")

    test = store['test']
    store.close()

    encoders = pickle.load(open(derived_data_path + "Encoders/all_encoders.p", "rb"))
    predictions = pd.read_csv(derived_data_path + 'predictions.csv', index_col=False)

    final_predictions = test.merge(predictions, on="bidder_id", how="left")
    final_predictions.fillna(0, inplace=True)

    final_predictions['bidder_id'] = encoders['id'].inverse_transform(final_predictions['bidder_id'])

    final_predictions[["bidder_id", "prediction"]].to_csv(derived_data_path + 'Predictions/' + name + '.csv', index=False)


def auction_level_model(derived_data_path):
    store = pd.HDFStore(derived_data_path + "derived_data.h5")
    bids = store['newbids']

    auction_bidder_level = bids.groupby(['auction', 'bidder_id']).agg({'bid_id': {'num_bids': 'size'}})
    auction_bidder_level.columns = auction_bidder_level.columns.droplevel()
    auction_bidder_level.reset_index(inplace=True)

    auction_bidder_level['pct_auction_bids'] = auction_bidder_level.groupby('auction').num_bids.transform(lambda x: x/x.sum()*100)
    auction_bidder_level['num_bids_auction'] = auction_bidder_level.groupby('auction').num_bids.transform('sum')
    auction_bidder_level['num_bidders_auction'] = auction_bidder_level.groupby('bidder_id').num_bids.transform('size')

    bids['num_bids_this_timestamp'] = bids.groupby('time').bid_id.transform('size')

    bids['num_bidders_this_auction'] = bids.groupby('auction').bidder_id.transform(lambda x : x.nunique())

    to_bidder_level = auction_bidder_level.groupby('bidder_id')

    new_bidder_level_stuff = to_bidder_level.pct_auction_bids.agg({'mean_pct_auction_bids': 'mean',
                                                                   'std_pct_auction_bids': 'std'})

    new_bidder_level_stuff.std_pct_auction_bids.fillna(-1, inplace=True)

    new_bidder_level_stuff['mean_num_bidders_auction'] = to_bidder_level.num_bidders_auction.agg('mean')
    new_bidder_level_stuff['mean_num_bids_auction'] = to_bidder_level.num_bids_auction.agg('mean')

    new_bidder_level_stuff['mean_bids_timestamp'] = bids.groupby('bidder_id').num_bids_this_timestamp.agg('mean')
    new_bidder_level_stuff['std_bids_timestamp'] = bids.groupby('bidder_id').num_bids_this_timestamp.agg('std')

    new_bidder_level_stuff.std_bids_timestamp.fillna(-1, inplace=True)

    return new_bidder_level_stuff

def auction_winner_stuff(derived_data_path):
    store = pd.HDFStore(derived_data_path + "derived_data.h5")
    bids = store['newbids']

    time_thresholds = [[1, 2],
                       [3, 4],
                       [5, 6]]



def bidder_level_model(derived_data_path):
    store = pd.HDFStore(derived_data_path + "derived_data.h5")
    bids = store['newbids']

    bids = bids.merge(train[["bidder_id", "outcome"]], on='bidder_id', how='left')

    # Time periods
    bids['time_period'] = bids.time.map(lambda x: 1 if x < 0.2 * 10**14 else 2)
    bids.loc[bids.time > 1.2 * 10**14, "time_period"] = 3

    bids = bids.join(store['simultaneous_df'])

    for var in ['simul_auction', 'simul_country', 'simul_device', 'simul_ip']:
        bids[var].fillna(0, inplace=True)

    # Stats by auction/user
    grouped = bids.groupby("bidder_id")
    bids['country'].fillna("NA", inplace=True)

    # Make some variables. Not pretty
    bidder_level = grouped.agg({'bid_id': {'num_bids': lambda x: len(x)},
                                'auction': {'num_auctions': lambda x: x.nunique()},
                                'merchandise': {'num_merchandise': lambda x: x.nunique(),
                                                'most_freq_merchandise': lambda x: x.value_counts().index[0]},
                                'device': {'num_device': lambda x: x.nunique(),
                                           'most_freq_device': lambda x: x.value_counts().index[0]},
                                'country': {'num_country': lambda x: x.nunique(),
                                            'most_freq_country': lambda x: x.value_counts().index[0]},
                                'ip': {'num_ip': lambda x: x.nunique(),
                                       'most_freq_ip': lambda x: x.value_counts().index[0]},
                                'url': {'num_url': lambda x: x.nunique(),
                                        'most_freq_url': lambda x: x.value_counts().index[0]},
                                'time': {'first_bid': lambda x: x.min(),
                                         'last_bid': lambda x: x.max(),
                                         'length_activity': lambda x: x.max() - x.min()},
                                'change_time': {'mean_change_time': np.mean,
                                                'std_change_time': np.std},
                                'first_bid': {'num_first_bid': lambda x: x.sum(),
                                              'pct_first_bid': lambda x: x.mean()},
                                'last_bid': {'num_winner': lambda x: x.sum(),
                                             'pct_winner': lambda x: x.mean()},
                                'same_bidder': {'num_outbid_self': lambda x: x.sum()},
                                'same_bidder_diff_time': {'num_outbid_self_diff_time': lambda x: x.sum()},
                                'time_period': {'num_periods': lambda x: x.nunique(),
                                                'primary_period': lambda x: x.value_counts().index[0]},
                                'simul_actions': {'max_simul_actions': lambda x: x.max(),
                                                  'mean_simul_actions': lambda x: x.mean()},
                                'simul_auction': {'max_simul_auction': lambda x: x.max(),
                                                  'mean_simul_auction': lambda x: x.mean()},
                                'simul_country': {'max_simul_country': lambda x: x.max(),
                                                  'mean_simul_country': lambda x: x.mean()},
                                'simul_device': {'max_simul_device': lambda x: x.max(),
                                                 'mean_simul_device': lambda x: x.mean()},
                                'simul_ip': {'max_simul_ip': lambda x: x.max(),
                                             'mean_simul_ip': lambda x: x.mean()},
                                'period': {'most_common_auction_type': lambda x: x.value_counts().index[0]},
                                'periods_seen': {'periods_seen': lambda x: x.mean()}})

    bidder_level.columns = bidder_level.columns.droplevel()
    bidder_level.head()

    ip_df = breakdown_ip_address(bidder_level['most_freq_ip'])

    other_bidder_level = auction_level_model(derived_data_path)

    store.put('bidder_level', pd.concat([bidder_level, ip_df,  other_bidder_level], axis=1))
    store.close()

if __name__ == '__main__':
    data_path = "C:/Users/P_Kravik/Desktop/Kaggle/Facebook Recruiting IV - Human or Robot/Data/"
    raw_data_path = data_path + "Raw/"
    derived_data_path = data_path + "Derived/"

    clean_data(raw_data_path, derived_data_path)
    save_to_hdf(derived_data_path)
    data_prep(derived_data_path)
    bidder_level_model(derived_data_path)
    print "done!"


