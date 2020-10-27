import numpy as np
np.random.seed(1337)

import pandas as pd
from scipy import signal
import scipy as sp
import tqdm 
import itertools
import os
import sys
import glob
import time
from pathlib import Path

import config_paper as config

hemisphere = config.constants['HEMISPHERES']
patient_ids = config.constants['PATIENT_IDS_PAPER']
decoder_days = config.constants['ECOG_DAYS_PAPER']
DATA_DIR = config.constants['DATA_DIR']
F_ECOG = config.constants['F_ECOG']
_MAX_TF_FREQ = config.constants['MAX_TF_FREQ'] # Max Hz for spectrograms
_NPERSEG = config.constants['NPERSEG']
_SCALING = config.constants['SCALING']


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



### EVENTS ###
def load_events_for_patient_id(patient_id, limb, rest=False):
    fname = "{}/events/events_{}_{}.csv".format(DATA_DIR, patient_id, limb)
    events_df = pd.read_csv(fname)
    events_df['patient_id'] = patient_id
    events_df['subject_id'] = 'S{:02d}'.format( patient_ids.index(patient_id) + 1)

    if not rest:
        events_df = events_df.query("mvmt != 'mv_0'")
    return events_df

def get_eventspans_from_events_ajile(events, window_before, window_after):
    """
    Eventspan: window_before (sec) --[Event]-- window_after (sec)
    """
    eventspans = events.copy()
    window_before = pd.Timedelta(window_before, unit='s')
    window_after = pd.Timedelta(window_after, unit='s')
    eventspans['start_time'] = eventspans['time'] - window_before 
    eventspans['end_time'] = eventspans['time'] + window_after
    eventspans['event_timespan'] = eventspans['end_time'] - eventspans['start_time']
    return eventspans

def rebalance_move_rest_dfs(move_df, rest_df):    
    move_df_resampled = []
    rest_df_resampled = []
    for day in move_df['day'].unique():
        # print(day)
        nrows_move_day = move_df.query('day == {}'.format(day)).shape[0] 
        nrows_rest_day = rest_df.query('day == {}'.format(day)).shape[0] 
        if nrows_rest_day >= nrows_move_day: # downsample rest_df for this day
            move_df_resampled.append( move_df.query('day == {}'.format(day)) )
            rest_df_resampled.append( rest_df.query('day == {}'.format(day)).sample(n=nrows_move_day) ) 
        else: # downsample move_df for this day... unusual!
            print("Unusual: Downsampling move_df from {} to {} for day {}".format(nrows_move_day, nrows_rest_day, day))
            move_df_resampled.append( move_df.query('day == {}'.format(day)).sample(n=nrows_rest_day) )
            rest_df_resampled.append( rest_df.query('day == {}'.format(day)) ) 
    move_df = pd.concat(move_df_resampled)
    rest_df = pd.concat(rest_df_resampled)
    # print("move_df", move_df.groupby(['day', 'mvmt']).count()['time'])
    # print("rest_df", rest_df.groupby(['day', 'mvmt']).count()['time'])
    return move_df, rest_df



### ECOG ###



### PREDICTION / DECODING ###
def get_ecog_for_timespan(cache_prefix, electrode_ids, 
                ecog_start_idx, ecog_end_idx, metadata_df):
    # Load ECoG from cache
    cache_file = "{}/{}_{}.npy".format(cache_prefix, ecog_start_idx, ecog_end_idx) 
    y_ecog_all = None
    if Path(cache_file).is_file():
        y_ecog_all = np.load(cache_file)
    
    # Zero out bad channels (from metadata) - might be redundant
    bad_channels = metadata_df['goodChanInds'].astype(int) == 0
    num_electrodes = y_ecog_all.shape[0]
    for idx in range(num_electrodes):
        if bad_channels[idx]:
            y_ecog_all[idx, :] *= 0.0
    return y_ecog_all
    
def get_ecog_features(y_ecog_all, f_ecog, electrode_ids, feature_mode):
    features = []
    feature_names = []
    
    # Multi-electrode spectrogram - many frequencies 
    if feature_mode == 'multi_sxx':
        for electrode_id in electrode_ids: # 
            y_ecog = y_ecog_all[electrode_id]
            f, t, Sxx = signal.spectrogram(y_ecog, fs=f_ecog, nperseg=_NPERSEG, scaling=_SCALING)

            if _MAX_TF_FREQ is not None:
                max_idx = sum(f <= _MAX_TF_FREQ)
                f = f[0:max_idx]
                Sxx = Sxx[0:max_idx, :]

            features.append(Sxx.ravel())
            feature_names.extend( ["e{}_f{:.1f}_t{:.1f}".format(electrode_id, x, y) for x,y in list(itertools.product(f, t))] )
        features = np.array(features).ravel()

    return features, feature_names

def get_ajile_train_test_data(patient_id, 
    electrode_ids, 
    mvti, 
    window_before, 
    window_after, 
    train_days, 
    test_days, 
    feature_mode):

    X_train, X_test, y_train, y_test = [], [], [], []
    feature_names = []
    days = train_days + test_days # Concat lists

    for day in days:
        print("Loading data for day", day)
        # ECoG event data caching
        cache_prefix = None
        cache_prefix = "{}/ecog_mvti_length/{}_{}/".format(DATA_DIR, patient_id, day)

        # Get Ecog
        metadata_fname = '{}/ecog_metadata/ecog_metadata_{}.csv'.format(DATA_DIR, patient_id)
        metadata_df = pd.read_csv(metadata_fname)

        # Get Events
        events = mvti[ mvti['day'] == day ]
        events.head()
        eventspans = get_eventspans_from_events_ajile(events, window_before, window_after)    
#         print(eventspans.head())

        # Get features
        Xs, ys, feature_names, event_idxs = get_Xs_ys_featnames(eventspans, patient_id, day, electrode_ids, metadata_df)
        
        # skip adding where the rare NaN shows up
        ok_markers = (~np.isnan(Xs).any(axis=1)).astype(bool)
        ok_idxs = [i for i in range(len(ys)) if ok_markers[i] ]
        Xs = Xs[ok_idxs]
        ys = ys[ok_idxs]
        event_idxs = event_idxs[ok_idxs]

        # nan: rebalance by day -- ASSUMES DATA SORTED MOVE-then-REST 
        n0 = np.sum( np.array(ys) == 'mv_0' )
        n1 = np.sum( np.array(ys) != 'mv_0' )
        print("Day {}: move {} vs rest {}".format(day, n1, n0))
        if n0 <= n1: # more move than rest, skip early items
            ndiff = n1-n0 
            Xs = Xs[ndiff:]
            ys = ys[ndiff:]
            event_idxs = event_idxs[ndiff:]
        else:
            ndiff = n0-n1 
            Xs = Xs[:-ndiff]
            ys = ys[:-ndiff]
            event_idxs = event_idxs[:-ndiff]
            
        # Add to train/test set
        if day in train_days:
            X_train.extend(Xs)
            y_train.extend(ys)
        if day in test_days:
            X_test.extend(Xs)
            y_test.extend(ys)
            feature_names = feature_names


    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, feature_names

def get_Xs_ys_featnames(eventspans, patient_id, day, electrode_ids, metadata_df):
    print(eventspans.columns)
    cache_prefix = "{}/ecog_mvti_length/{}_{}/".format(DATA_DIR, patient_id, day)
    ecog_d =  None

    Xs = []
    ys = []
    event_idxs = []

    # Sort before processing - move events before rest
    eventspans['rest'] = (eventspans['mvmt'] == 'mv_0') + 0 # 0:move 1:rest  
    eventspans.sort_values(by='rest', inplace=True)

    for idx, row in tqdm.tqdm(eventspans.iterrows()):
        event = row['mvmt']
        start_time = row['start_time']
        end_time = row['end_time']
        
        ecog_start_idx = row['ecog_start_idx_mvti']
        ecog_end_idx = row['ecog_end_idx_mvti']

        try:
            y_ecog_all = get_ecog_for_timespan(cache_prefix, electrode_ids, 
                    ecog_start_idx, ecog_end_idx, metadata_df)
            features, feature_names = get_ecog_features(y_ecog_all, F_ECOG, 
                    electrode_ids, feature_mode='multi_sxx')
            # print(event, len(features), len(feature_names))
            Xs.append(features)
            ys.append(event)
            event_idxs.append("{}_{}".format(ecog_start_idx, ecog_end_idx))

        except Exception as e:
            print("Exception!", e)

    Xs, ys = np.array(Xs), np.array(ys)
    event_idxs = np.array(event_idxs)
    # assert Xs.shape[0] > 0
    # assert ys.shape[0] > 0
    return Xs, ys, feature_names, event_idxs


# def get_ajile_train_test_data(patient_id, 
#     electrode_ids, 
#     mvti, 
#     window_before, 
#     window_after, 
#     train_days, 
#     test_days, 
#     feature_mode):

#     X_train, X_test, y_train, y_test = [], [], [], []
#     feature_names = []
#     days = train_days + test_days # Concat lists

#     for day in days:
#         print("Loading data for day", day)
#         # ECoG event data caching
#         cache_prefix = None
#         cache_prefix = "{}/ecog_mvti_length/{}_{}/".format(DATA_DIR, patient_id, day)

#         # Get Ecog
#         metadata_fname = '{}/ecog_metadata/ecog_metadata_{}.csv'.format(DATA_DIR, patient_id)
#         metadata_df = pd.read_csv(metadata_fname)

#         # Get Events
#         events = mvti[ mvti['day'] == day ]
#         events.head()
#         eventspans = get_eventspans_from_events_ajile(events, window_before, window_after)    
# #         print(eventspans.head())

#         # Get features
#         Xs, ys, feature_names = get_Xs_ys_featnames(eventspans, patient_id, day, electrode_ids, metadata_df)
#         if day in train_days:
#             X_train.extend(Xs)
#             y_train.extend(ys)
#         if day in test_days:
#             X_test.extend(Xs)
#             y_test.extend(ys)
#             feature_names = feature_names

#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)
#     return X_train, X_test, y_train, y_test, feature_names


def get_forest_feature_importance(clf, feature_names, viz=True): # TODO later
    # Feature importance
    featureImportances = clf.feature_importances_
    featureImportances = 100.0 * (featureImportances / featureImportances.sum()) #Normalize 
    topFeatures = pd.DataFrame(featureImportances, columns=['Importance'], index=feature_names) 
    topFeatures = topFeatures.sort_values(by='Importance',  ascending=False)
    if viz:
        topFeatures.reset_index().plot()
        plt.xlabel("Feature Index (Sorted) ")
        plt.ylabel("Feature Importance (Normalized)")
    return topFeatures


def get_electrode_loadings(topFeaturesAll):
    # Work with either PSD or Context Features
    topFeatures = topFeaturesAll[[ 'e' == x[:1] for x in topFeaturesAll.index ]] # only electode loadings
    topFeatures['Importance'] = topFeatures['Importance']*100/topFeatures['Importance'].sum()
    topFeatures.head(10)

    # Aggregate loadings by electrode_id
    topFeatures['electrode_id'] = topFeatures.index.to_series().apply(lambda x: x.split('_')[0].replace('e',''))
    topFeatures['electrode_id'] = topFeatures['electrode_id'].astype(int)
    loadings = topFeatures.groupby('electrode_id').sum()
    loadings['electrode_id'] = loadings.index
    loadings.reset_index(drop=True, inplace=True)
    loadings.head()
    return loadings

def remove_na_train_test(X_train, X_test, y_train, y_test):
    print("Original:", [np.array(x).shape for x in [X_train, X_test, y_train, y_test]])
    # Save old "all" data just in case - only run once!
    X_train_all = X_train
    X_test_all = X_test
    y_train_all = y_train
    y_test_all = y_test

    # Subset to rows without NaNs
    # https://stackoverflow.com/questions/11453141/how-to-remove-all-rows-in-a-numpy-ndarray-that-contain-non-numeric-values
    X_train_clean = X_train[~np.isnan(X_train).any(axis=1)]
    y_train_clean = y_train[~np.isnan(X_train).any(axis=1)]
    X_train_clean.shape, y_train_clean.shape

    X_test_clean = X_test[~np.isnan(X_test).any(axis=1)]
    y_test_clean = y_test[~np.isnan(X_test).any(axis=1)]
    X_test_clean.shape, y_test_clean.shape

    X_train = X_train_clean
    X_test = X_test_clean
    y_train = y_train_clean
    y_test = y_test_clean
    print("After NAs removed:", [np.array(x).shape for x in [X_train, X_test, y_train, y_test]])

    return X_train, X_test, y_train, y_test

def eval_classifier_fit(clf, X_train, y_train, X_test, y_test, 
    reports=['accuracy', 'per_class', 'confusion_matrix'],
    verbose=True):
    return_dict = {}
    # Training
    y_hat = clf.predict(X_train)

    if 'accuracy' in reports:
        return_dict['train_accuracy'] = accuracy_score(y_train, y_hat)
        if verbose:
            print("Training accuracy:", return_dict['train_accuracy'])

    if 'per_class' in reports:
        print('classification_report(y_train, y_hat):')
        return_dict['classification_report_train'] = classification_report(y_train, y_hat, output_dict=True)
        print(classification_report(y_train, y_hat))

    if 'confusion_matrix' in reports:
        print('confusion_matrix(y_test, y_hat):')
        return_dict['confusion_matrix_train'] = None
        try:
            return_dict['confusion_matrix_train'] = confusion_matrix(y_train, y_hat)
            print(return_dict['confusion_matrix_train'])
        except Exception as e:
            print("Exception: ", e)

    # Testing
    y_hat = clf.predict(X_test)
    if 'accuracy' in reports:
        return_dict['test_accuracy'] = accuracy_score(y_test, y_hat)
        if verbose:
            print("Test accuracy:", return_dict['test_accuracy'])

    if 'per_class' in reports:
        print('classification_report(y_test, y_hat):')
        return_dict['classification_report_test'] = classification_report(y_test, y_hat, output_dict=True)
        print(classification_report(y_test, y_hat))

    if 'confusion_matrix' in reports:
        print('confusion_matrix(y_test, y_hat):')
        return_dict['confusion_matrix_test'] = None
        try:
            return_dict['confusion_matrix_test'] = confusion_matrix(y_test, y_hat)
            print(return_dict['confusion_matrix_test'])
        except Exception as e:
            print("Exception: ", e)

    return return_dict
