"""
Back end for Decoder.ipynb [see instructions there]
BASH snippets 
Run in directory where decoder_cli_paper.py exists

# Test with: 
python decoder_cli_paper.py --patient_id a0f66459 --testing

# Run with:
PATIENT_IDS="a0f66459" # Test with one patient
PATIENT_IDS="ffb52f92" # Test with one patient
PATIENT_IDS="a0f66459 c95c1e82 cb46fd46 ffb52f92 fcb01f7a b4ac1726 f3b79359 ec761078 f0bbc9a9 abdb496b ec168864 b45e3f7b"
OUTDIR='reports/'

mkdir -p ${OUTDIR}
for PATIENT_ID in $PATIENT_IDS
do
  for LIMB in l_wrist r_wrist 
  do
    echo $PATIENT_ID ${LIMB}
    python decoder_cli_paper.py --patient_id ${PATIENT_ID} --limb ${LIMB} --hyperopt > reports/${PATIENT_ID}_${LIMB}.out 2>&1
  done
done

# Interactive debug w/ full data
ipython -i decoder_cli_paper.py -- --patient_id ffb52f92 --limb l_wrist --hyperopt
ipython -i decoder_cli_paper.py -- --patient_id c95c1e82 --limb l_wrist --hyperopt



"""
SEED=1337
import numpy as np
np.random.seed(SEED)

import pandas as pd
import scipy as sp
import sys, os
import argparse
import glob
import pickle
import tqdm
import itertools
from pathlib import Path
from scipy import signal

# ML
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# custom
import paper_utils as paperu
import config_paper as config


hemisphere = config.constants['HEMISPHERES']
patient_ids = config.constants['PATIENT_IDS_PAPER']
decoder_days = config.constants['ECOG_DAYS_PAPER']
DATA_DIR = config.constants['DATA_DIR']
# _MAX_TF_FREQ = config.constants['MAX_TF_FREQ'] # Max Hz for spectrograms
# _NPERSEG = config.constants['NPERSEG']
# _SCALING = config.constants['SCALING']
# F_ECOG = config.constants['F_ECOG']


### ARGS ###
parser = argparse.ArgumentParser(description='Decoder for regions')
parser.add_argument('--patient_id',  metavar='p', type=str)
parser.add_argument('--testing',  dest='testing', action='store_true')  
parser.add_argument('--limb',  metavar='l', type=str, default='r_wrist')
# parser.add_argument('--feature_mode',  metavar='f', type=str, default='multi_lfb_hfb')
parser.add_argument('--max_jobs',  metavar='j', type=int, default=4)
parser.add_argument('--hyperopt',  dest='hyperopt', action='store_true')  
args = parser.parse_args()
print("Parsed arguments:", args)
if None in [args.patient_id, args.limb]:
  print("Missing args")
  sys.exit(-1)

patient_id = args.patient_id
limb = args.limb
_TESTING = args.testing
# _FEATURE_MODE = args.feature_mode
HYPOPT = args.hyperopt
MAX_JOBS=args.max_jobs


print('<patient_id, limb>', patient_id, limb, "TESTING" if _TESTING else '')



# OVERRIDES -- Common
_PREDICTORS = [['classifier'], ['regressor']][0]
_ANALYSIS = ['move_vs_rest', 'metadata', 'multitrack'][0]
_FEATURE_MODE = ['single_psd', #0
                 'multi_psd', #1
                 'multi_psd_lfb_hfb', #2 
                 'multi_sxx', #3
                 'multi_sxx_lfb_hfb', #4
                 # 'multi_psd_fooof' #5
                ][3]
_CONTEXT = None if _ANALYSIS is 'move_vs_rest' else 'metadata' 
_CLASSIFICATION_ALGORITHMS = [
  'RandomForestClassifier',
]

HYPOPT_ITER = 20
HYPOPT_CV = 5
if _TESTING:
  HYPOPT_ITER = 1
  HYPOPT_CV = 2


#Setup ECoG
ecog_days = config.constants['ECOG_DAYS_PAPER']
train_days = ecog_days[patient_id][:-1]
test_days = [ ecog_days[patient_id][-1] ]
print("patient_id, <train_days, test_days>", patient_id, train_days, test_days)
all_days = train_days + test_days
day_min, day_max = min(all_days), max(all_days)
electrodes_all = np.arange(64) # Fixed # for now
_WINDOW_BEFORE, _WINDOW_AFTER  = 0.5, 0.5 
halfwindow_secs = _WINDOW_BEFORE + _WINDOW_AFTER # for nearby event removal


### Load events data
events_df = paperu.load_events_for_patient_id(patient_id, limb, rest=True)
events_df = events_df.query('day >= {} and day <= {}'.format(day_min, day_max))
print('Filter: days min/max', events_df.shape, day_min, day_max)

events_df['bimanual'] = (events_df['other_overlap_15pm'] > 6) + 0 # TODO: Redundant
move_df = events_df.query("mvmt != 'mv_0'")
move_df = move_df.query('bimanual == 0') # Unimanual only
print('Filter: bimanual15v2', move_df.shape)

rest_df = events_df.query("mvmt == 'mv_0'")
move_df, rest_df = paperu.rebalance_move_rest_dfs(move_df, rest_df)
mvti = pd.concat([ move_df, rest_df ])
print("Rebalanced", mvti.groupby(['day', 'mvmt']).count()['time'])

if _TESTING:
    mvti = mvti.sample(frac=0.1) # TESTING!
    print("TESTING - Reduced data to shape: ", mvti.shape)

if _ANALYSIS == 'move_vs_rest':
    colnames = [
    'day', 
    'time', 
    'event_timestamp', 
    'mvmt',
    'event_frame_idx',
    'ecog_start_idx_mvti',
    'ecog_start_idx_full',
    'ecog_end_idx_mvti',
    'ecog_end_idx_full',
    ]
    mvti = mvti.loc[:, colnames]

mvti = mvti.reset_index(drop=True)
print(mvti.groupby(['day', 'mvmt']).count()['time'])

# ## Create events / eventspan data 
# ### Multiple-day features
electrode_ids = electrodes_all
X_train, X_test, y_train, y_test, feature_names = \
    paperu.get_ajile_train_test_data(patient_id, electrode_ids, mvti, 
            _WINDOW_BEFORE, _WINDOW_AFTER, 
            train_days=train_days, 
            test_days=test_days, 
            feature_mode=_FEATURE_MODE)

X_train, X_test, y_train, y_test = paperu.remove_na_train_test(X_train, X_test, y_train, y_test)






### Common: Classify -- Categorical Variable ###
def classification_biolerplate(X_train, y_train_binned, X_test, y_test_binned, 
  algorithm, hyperparam_opt=False):
  # Fit model
  if algorithm == 'RandomForestClassifier':
    model = RandomForestClassifier(n_estimators=100,
                 max_depth=5, 
                 class_weight="balanced",
                 random_state=SEED,
                 n_jobs=MAX_JOBS,
                 oob_score=True)
    param_distributions = {"n_estimators": randint(50, 250), 
               "max_depth": randint(3, 15)}

  if algorithm in ['RandomForestClassifier']: # Trees don't need scaling
    if hyperparam_opt:
      rnd_search_cv = RandomizedSearchCV(model, param_distributions, 
                                         n_iter=HYPOPT_ITER, verbose=2, 
                                         cv=HYPOPT_CV, random_state=SEED)
      rnd_search_cv.fit(X_train, y_train_binned.ravel()) # Fit best model
      clf = rnd_search_cv.best_estimator_
      acc_val = clf.oob_score_ 
      best_params = rnd_search_cv.best_params_
      print("best_params_", rnd_search_cv.best_params_)
    else: 
      clf = model.fit(X_train, y_train_binned.ravel())
      acc_val = clf.oob_score_ 
      best_params = None

  else: # With scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.astype(np.float32))
    if hyperparam_opt:
      rnd_search_cv = RandomizedSearchCV(model, param_distributions, 
                                         n_iter=HYPOPT_ITER, verbose=2, 
                                         cv=HYPOPT_CV, random_state=_SEED)
      rnd_search_cv.fit(X_train_scaled, y_train_binned.ravel())
      clf = rnd_search_cv.best_estimator_
      acc_val = rnd_search_cv.best_score_
      best_params = rnd_search_cv.best_params_
      print("best_params_", rnd_search_cv.best_params_)
    else: # No HypOpt
      clf = model.fit(X_train_scaled, y_train_binned.ravel())
      acc_val = np.nan
      best_params = None

  # Make return_dict
  return_dict = paperu.eval_classifier_fit(clf, X_train, y_train_binned, 
          X_test, y_test_binned, 
          # reports=['accuracy'], 
          reports=['accuracy', 'per_class', 'confusion_matrix'],
          verbose=False)
  return_dict['val_accuracy'] = acc_val
  return_dict['best_params'] = best_params
  return return_dict, clf

### Identity: Classify Categorical Variables ###
model_dict = {}
decoder_report = []
if _ANALYSIS == 'move_vs_rest':
  for algorithm in _CLASSIFICATION_ALGORITHMS:

    print (algorithm, _ANALYSIS)
    return_dict, clf = classification_biolerplate(X_train, y_train, X_test, y_test, 
      algorithm, hyperparam_opt=HYPOPT)

    # Report
    n_bins = 2
    colname = 'm_vs_r'
    dict_key = '{}_{}_{}'.format('identity', colname, algorithm)
    model_dict[dict_key] = [clf, return_dict, feature_names, _FEATURE_MODE,
      X_train, X_test, y_train, y_test]
    decoder_report.append([colname, 
                           algorithm, 
                           return_dict['train_accuracy'],
                           return_dict['val_accuracy'],
                           return_dict['test_accuracy'],
                          ])

  decoder_report = pd.DataFrame(decoder_report)
  decoder_report.columns = ['predicted_variable', 'strategy', 'train', 'val', 'test']
  print(_FEATURE_MODE)
  print(decoder_report)
  fname = 'reports/{}_{}_{}_{}.csv'.format(patient_id, limb, 'identity', 'classify')
  decoder_report.to_csv(fname)
  fname = fname.replace('.csv', '.pickle')
  with open(fname, 'wb') as f_handle:
      pickle.dump(model_dict, f_handle)


