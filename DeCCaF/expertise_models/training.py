import os
import itertools
#hi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from sklearn import metrics
from aequitas.group import Group

from autodefer.models import haic
from autodefer.utils import thresholding as t, plotting

import pickle

sns.set_style('whitegrid')


cfg_path ='cfg.yaml'

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

RESULTS_PATH = cfg['results_path'] + '/'
MODELS_PATH = cfg['models_path']  + '/'

data_cfg_path = '../data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']


os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

with open(cfg['metadata'], 'r') as infile:
    metadata = yaml.safe_load(infile)

LABEL_COL = metadata['data_cols']['label']
PROTECTED_COL = metadata['data_cols']['protected']
CATEGORICAL_COLS = metadata['data_cols']['categorical']
TIMESTAMP_COL = metadata['data_cols']['timestamp']

SCORE_COL = metadata['data_cols']['score']
BATCH_COL = metadata['data_cols']['batch']
ASSIGNMENT_COL = metadata['data_cols']['assignment']
DECISION_COL = metadata['data_cols']['decision']

EXPERT_IDS = metadata['expert_ids']

print(cfg['train_paths'])

TRAIN_ENVS = {
    tuple(exp_dir.split('#')): {
        'train': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/train.parquet'),
        'batches': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/batches.parquet'),
        'capacity': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/capacity.parquet'),
    }
    for exp_dir in os.listdir(cfg['train_paths']['environments'])
    if os.path.isdir(cfg['train_paths']['environments']+exp_dir)
}

# DEFINING Lambda (fp cost) ---------------------------------------------------------------------------------

with open(f'../ml_model/model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

ml_model_threshold = ml_model_properties['threshold']
ml_model_recall = 1 - ml_model_properties['fnr']
ml_model_fpr_diff = ml_model_properties['disparity']
ml_model_fpr = ml_model_properties['fpr']

#Our defined lambda
THEORETICAL_FP_COST = -ml_model_threshold / (ml_model_threshold - 1)

# Training our Expertise Model. A user can train this model under various training conditions, defined in testbed_train_generation.py
VAL_ENVS = dict()
VAL_X = None
RMAs = dict()
for env_id in TRAIN_ENVS:
    batch_id, capacity_id = env_id
    models_dir = f'{MODELS_PATH}{batch_id}_{capacity_id}/'
    os.makedirs(models_dir, exist_ok=True)

    train_with_val = TRAIN_ENVS[env_id]['train']
    train_with_val = train_with_val.copy().drop(columns=BATCH_COL)

    #Possibly subsample here


    is_val = (train_with_val[TIMESTAMP_COL] == 6)
    train_with_val = train_with_val.drop(columns=TIMESTAMP_COL)
    train = train_with_val[~is_val].copy()
    val = train_with_val[is_val].copy()

    RMAs[env_id] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{models_dir}',
    )

    RMAs[env_id].fit(
        train=train,
        val=val,
        categorical_cols=CATEGORICAL_COLS, score_col=SCORE_COL,
        decision_col=DECISION_COL, ground_truth_col=LABEL_COL, assignment_col=ASSIGNMENT_COL,
        hyperparam_space=cfg['human_expertise_model']['hyperparam_space'],
        n_trials=cfg['human_expertise_model']['n_trials'],
        random_seed=cfg['human_expertise_model']['random_seed'], 
        CAT_DICT = cat_dict
    )

RMAs = dict()

for train_env in os.listdir(MODELS_PATH):
    RMAs[train_env] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{MODELS_PATH}{train_env}/',
    )
    calibrator_path = f'{MODELS_PATH}{train_env}/calibrator.pickle'
    RMAs[train_env].load(CATEGORICAL_COLS, SCORE_COL, ASSIGNMENT_COL, calibrator_path, cat_dict)

from sklearn.calibration import IsotonicRegression, calibration_curve


data = pd.read_parquet('../data/BAF_deployment_score.parquet')
orig = pd.read_parquet('../data/BAF.parquet')

val = data.loc[orig['month'] == 6]
val_preds = pd.read_parquet('/mnt/home/jean.alves/Alert_review_new/experts/expert_info/deployment_predictions.parquet').loc[orig['month'] == 6]
temp = val.copy()
temp['labels'] = val['fraud_bool']
val = val.drop(columns = 'fraud_bool')
Classes = np.array(['fn', 'fp', 'tn', 'tp'])


def get_outcome(label, pred):
    if pred == 1:
        if label == 1:
            o = 'tp'
        elif label == 0:
            o = 'fp'
    elif pred == 0:
        if label == 1:
            o = 'fn'
        elif label == 0:
            o = 'tn'
    return o


def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype != 'category':
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
        elif new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

val_res = dict()
if os.path.isfile('val_prev_bal_ce.pkl'):
    with open('val_prev_bal_ce.pkl', 'rb') as fp:
        val_res = pickle.load(fp)
else:
    for env_id in os.listdir(MODELS_PATH):
        val = pd.read_parquet(f'../testbed/train_alert/{env_id.split("_")[0]}#{env_id.split("_")[1]}/train.parquet').loc[orig['month'] == 6]
        val = val.loc[val['assignment'] != 'model#0']
        curves = dict()
        
        model = RMAs[env_id]

        outcomes = val.apply(lambda x: get_outcome(label=x['fraud_bool'], pred=x['decision']),
                axis=1,
        )
        
        val = val.drop(columns = ['fraud_bool', 'decision', 'batch', 'month'])

        val = cat_checker(val, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)

        pred_proba = model.expert_model.predict_proba(val)
            
        val_res[env_id] = {
                            'total_pred_proba': pred_proba,
                            'total_outcomes': outcomes
                            }
    #with open('val_prev_bal_ce.pkl', 'wb') as fp:
     #   pickle.dump(val_res, fp)



calibrators = dict()
if os.path.isfile('calibrators.pkl'):
    with open('calibrators.pkl', 'rb') as fp:
        calibrators = pickle.load(fp)
else:
    for env_id in os.listdir(MODELS_PATH):

        calibrator_fp = IsotonicRegression(out_of_bounds = 'clip').fit(val_res[env_id]['total_pred_proba'][:, Classes == 'fp'], (val_res[env_id]['total_outcomes'] == 'fp').astype(int))
        calibrator_fn = IsotonicRegression(out_of_bounds = 'clip').fit(val_res[env_id]['total_pred_proba'][:, Classes == 'fn'], (val_res[env_id]['total_outcomes'] == 'fn').astype(int))   
        calibrators[env_id] = {
                            'fp': calibrator_fp,
                            'fn': calibrator_fn
                            }
    with open('calibrators.pkl', 'wb') as fp:
        pickle.dump(calibrators, fp)




test = pd.read_parquet('../testbed/test/test.parquet')
test_expert_pred = pd.read_parquet('../testbed/test/test_expert_pred.parquet')


test = cat_checker(test, data_cfg['data_cols']['categorical'], cat_dict)
test['assignment'] = 'blank'

test_y = test['fraud_label']
test = test.drop(columns = ['fraud_label', 'month'])
roc_curves = dict()

test_expert_pred['label'] = test_y

if os.path.isfile('calibrated_predictions.pkl'):
    with open('calibrated_predictions.pkl', 'rb') as fp:
        roc_curves = pickle.load(fp)
else:
    for env_id in os.listdir(MODELS_PATH):
        print(env_id)
        curves = dict()
        total_outcomes = []
        total_pred_proba_fp = []
        total_pred_proba_fn = []
        calibrator_fp = calibrators[env_id]['fp']
        calibrator_fn = calibrators[env_id]['fn']
        i=0
        for expert in EXPERT_IDS['human_ids']:
            model = RMAs[env_id]

            test['assignment'] = expert

            outcomes = test_expert_pred.apply(lambda x: get_outcome(label=x['label'], pred=x[expert]),
                axis=1,
            )   

            total_outcomes.append(outcomes.to_numpy().squeeze())

            test = cat_checker(test, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)
            pred_proba = model.expert_model.predict_proba(test)

            pred_proba_fp = pred_proba[:, Classes == 'fp'].squeeze()
            pred_proba_fn = pred_proba[:, Classes == 'fn'].squeeze()

            pred_proba_fp = calibrator_fp.transform(pred_proba_fp)
            pred_proba_fn = calibrator_fn.transform(pred_proba_fn)

            total_pred_proba_fp.append(pred_proba_fp)
            total_pred_proba_fn.append(pred_proba_fn)

           
            curves[expert]= {'fp': pred_proba_fp,
                             'fn': pred_proba_fn}
            i+=1
            
        
        total_pred_proba_fp = np.concatenate(total_pred_proba_fp)
        total_pred_proba_fn = np.concatenate(total_pred_proba_fn)
        
        total_outcomes = np.concatenate(total_outcomes)
       
        curves['all']= {
                            'total_fp': total_pred_proba_fp,
                            'total_fn': total_pred_proba_fn,
                            'total_outcomes': total_outcomes
                            }
        roc_curves[env_id] = curves
    with open('calibrated_predictions.pkl', 'wb') as fp:
        pickle.dump(roc_curves, fp)