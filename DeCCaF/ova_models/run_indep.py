import pandas as pd
import numpy as np
import os
import yaml
import hpo_ova_indep
import pickle

train_path = '../testbed/train_alert'
exp_pred = pd.read_parquet('../experts/expert_info/deployment_predictions.parquet')
experts = exp_pred.columns.drop('model#0')

data_cfg_path = '../data/dataset_cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

for train in os.listdir(train_path):
    train_set = pd.read_parquet(train_path  + f'/{train}/train.parquet')
    name = train.split('#')[0] + '_' + train.split('#')[1]
    for expert in experts:
        train_exp = train_set.loc[train_set['assignment'] == expert]
        train_exp = cat_checker(train_exp, CATEGORICAL_COLS, cat_dict)

        val_exp = train_exp.loc[train_exp['month'] == 6]
        train_exp = train_exp.loc[train_exp['month'] != 6]
        
        #train_w = train_exp['fraud_bool'].replace([0,1],[0.057,1])
        #val_w = val_exp['fraud_bool'].replace([0,1],[0.057,1])

        train_w = train_exp['fraud_bool'].replace([0,1],[1,1])
        val_w = val_exp['fraud_bool'].replace([0,1],[1,1])

        train_x = train_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])
        val_x = val_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])

        train_y = (train_exp['decision'] == train_exp['fraud_bool']).astype(int)
        val_y = (val_exp['decision'] == val_exp['fraud_bool']).astype(int)

        if not (os.path.exists(f'./models/{name}/{expert}/')):
            os.makedirs(f'./models/{name}/{expert}/')
        
        if not (os.path.exists(f'./models/{name}/{expert}/best_model.pickle')):
            opt = hpo_ova_indep.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'./models/{name}/{expert}/')
            opt.initialize_optimizer(CATEGORICAL_COLS, 25)


test = pd.read_parquet('../testbed/test/test.parquet')
test_expert_pred = pd.read_parquet('../testbed/test/test_expert_pred.parquet')

test = test.drop(columns = ['fraud_label', 'month'])

preds = dict()

for env in os.listdir('./models'):
    table = pd.DataFrame(index = test.index, columns = os.listdir(f'./models/{env}'))
    for expert in os.listdir(f'./models/{env}'):
        with open(f"./models/{env}/{expert}/best_model.pickle", "rb") as input_file:
            model = pickle.load(input_file)
        
        table.loc[:, expert] = model.predict_proba(test)[:,1]
    
    preds[env] = table

with open(f"ova_predictions.pkl", "wb") as input_file:
    pickle.dump(preds, input_file)

        



