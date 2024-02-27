# %%
import pandas as pd
import yaml
import numpy as np
import hpo_wce
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path





data_cfg_path = (Path(__file__).parent/'../data/dataset_cfg.yaml').resolve()
cfg_path = Path(__file__).parent/'cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)



cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)


# DATA LOADING -------------------------------------------------------------------------------------
scens = os.listdir('../data/alerts/')
for scen in scens:
    scen = scen.split('.parquet')[0]
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    data = pd.read_parquet(f'../data/alerts/{scen}.parquet')

    LABEL_COL = data_cfg['data_cols']['label']
    TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
    PROTECTED_COL = data_cfg['data_cols']['protected']
    CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

    data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

    train = data.loc[(data["month"] > 2) & (data["month"] < 6)]
    val = data.loc[data["month"] == 6]

    X_train = train.drop(columns = ['fraud_bool','model_score','month'])
    y_train = train['fraud_bool']

    X_val = val.drop(columns = ['fraud_bool','model_score','month']) 
    y_val = val['fraud_bool']

    for cost in cfg['costs']:
        if sub and (cost not in cfg['run_sub']):
            continue
        scen_c = scen + f'-l_{cost}'
        w_train = y_train.replace([0,1],[cost,1])
        w_val = y_val.replace([0,1],[cost,1])

        p_train = (y_train*w_train).sum()/(w_train.sum())
        p_val = (y_val*w_val).sum()/(w_val.sum())

        init_train = np.log((p_train)/(1-p_train))
        init_val = np.log((p_val)/(1-p_val))
        n = 0
        for param_space_dic in os.listdir('./param_spaces/'):
            with open('./param_spaces/' + param_space_dic, 'r') as infile:
                param_space = yaml.safe_load(infile)

            for initial in np.arange(init_train, init_train + 2, 0.2):
                param_space['init_score'] = initial
                os.makedirs(f'./models/{scen_c}/models', exist_ok=True)
                
                if not (os.path.exists(f'./models/{scen_c}/models/model_{n}')):
                    opt = hpo_wce.HPO(X_train,X_val,y_train,y_val,w_train,w_val, parameters = param_space, method = 'TPE', path = f"./models/{scen_c}/models/model_{n}")
                    opt.initialize_optimizer(CATEGORICAL_COLS, cfg['n_jobs'])
                    n +=1
                else:
                    print('model is trained')
                    n +=1

        models_path = f'./models/{scen_c}/models/'

        Trials = []

        for model in os.listdir(models_path):
            study = int(model.split('_')[-1])
            with open(models_path + model + '/history.yaml', 'r') as infile:
                param_hist = yaml.safe_load(infile)

            with open(models_path + model + '/config.yaml', 'r') as infile:
                conf = yaml.safe_load(infile)
            
            temp = pd.DataFrame(param_hist)
            temp['study'] = study
            temp['max_depth_max'] = conf['params']['max_depth']['range'][1]
            Trials.append(temp)

        Trials = pd.concat(Trials)
        Trials = Trials.reset_index(drop = True)
        Trials['study'] = Trials['study'].astype(int)
        a = Trials

        selec_ix = a.loc[a['ll'] == a['ll'].min(),'study'].to_numpy()[0]

        print(selec_ix)

        selected_model_path = f'./models/{scen_c}/models/model_{selec_ix}'

        with open(f'{selected_model_path}/best_model.pickle', 'rb') as infile:
            model = pickle.load(infile)

        with open(f'{selected_model_path}/config.yaml', 'r') as infile:
            model_cfg = yaml.safe_load(infile)

        CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

        test = data.loc[data["month"] == 7]

        X_test = test.drop(columns = ["month",'model_score', "fraud_bool"])
        y_test = test["fraud_bool"]
        w_test = y_test.replace([0,1],[cost,1])

        p_test = (y_test*w_test).sum()/(w_test.sum())
        init_test = np.log((p_test)/(1-p_test))

        selected_model = dict()
        init_score = model_cfg['init_score']
        selected_model['init_score'] = float(init_score)
        selected_model['threshold'] = 0.5

        model_preds = pd.Series(output(X_train,model, init_score) >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_train, model_preds).ravel()
        avg_cost_model = (cost*fp + fn)/(tn+fp+fn+tp)

        selected_model['fpr_train'] = float(fp/(fp+tn))
        selected_model['fnr_train'] = float(fn/(fn+tp))
        selected_model['prev_train'] = float(y_train.mean())
        selected_model['cost_train'] = float(avg_cost_model)

        tn, fp, fn, tp = confusion_matrix(y_train, np.ones(len(y_train))).ravel()
        avg_cost_full_rej = (cost*fp + fn)/(tn+fp+fn+tp)

        print(f"Training Set -- Model: {avg_cost_model:.3f}. Rejecting all: {avg_cost_full_rej:.3f}")

        model_preds = pd.Series(output(X_val,model, init_score) >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_val, model_preds).ravel()
        avg_cost_model = (cost*fp + fn)/(tn+fp+fn+tp)

        selected_model['fpr_val'] = float(fp/(fp+tn))
        selected_model['fnr_val'] = float(fn/(fn+tp))
        selected_model['prev_val'] = float(y_val.mean())
        selected_model['cost_val'] = float(avg_cost_model)

        tn, fp, fn, tp = confusion_matrix(y_val, np.ones(len(y_val))).ravel()
        avg_cost_full_rej = (cost*fp + fn)/(tn+fp+fn+tp)

        print(f"Val Set -- Model: {avg_cost_model:.5f}. Rejecting all: {avg_cost_full_rej:.5f}")

        model_preds = pd.Series(output(X_test,model, init_score) >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, model_preds).ravel()
        avg_cost_model = (cost*fp + fn)/(tn+fp+fn+tp)

        selected_model['fpr_test'] = float(fp/(fp+tn))
        selected_model['fnr_test'] = float(fn/(fn+tp))
        selected_model['prev_test'] = float(y_test.mean())
        selected_model['cost_test'] = float(avg_cost_model)

        tn, fp, fn, tp = confusion_matrix(y_test, np.ones(len(y_test))).ravel()
        avg_cost_full_rej = (cost*fp + fn)/(tn+fp+fn+tp)

        print(f"Test Set -- Model: {avg_cost_model:.5f}. Rejecting all: {avg_cost_full_rej:.5f}")

        os.makedirs(f'./selected_models/{scen_c}', exist_ok=True)

        with open(f'./selected_models/{scen_c}/best_model.pickle', 'wb') as outfile:
            pickle.dump(model, outfile)

        with open(f'./selected_models/{scen_c}/model_properties.yaml', 'w') as outfile:
            yaml.dump(selected_model, outfile)