# %%
import pandas as pd
import numpy as np
import os
import yaml
import hpo
import pickle

data_cfg_path = '../data/dataset_cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

with open('cfg.yaml', 'r') as infile:
    cfg = yaml.safe_load(infile)

CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data


scens = os.listdir('../data/alerts/')
costs = cfg['costs']
for scen in scens:
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs:
        if sub and (l not in cfg['run_sub']):
            continue
        scen = scen.split('.parquet')[0]
        expert_ids_path = f'../experts/teams/{scen}-l_{l}/expert_info/expert_ids.yaml'

        with open(expert_ids_path, 'r') as infile:
            EXPERT_IDS = yaml.safe_load(infile)

        experts = EXPERT_IDS['human_ids'] 

        for train in os.listdir(f'../testbed/testbed/{scen}-l_{l}/train_alert'):
            train_set = pd.read_parquet(f'../testbed/testbed/{scen}-l_{l}/train_alert/{train}/train.parquet')
            train_set = train_set.loc[train_set["assignment"] != 'model#0']
            for expert in experts:
                print(f'Fitting start for {train}, expert {expert}')
                train_exp = train_set.loc[train_set['assignment'] == expert]
                train_exp = cat_checker(train_exp, CATEGORICAL_COLS, cat_dict)

                val_exp = train_exp.loc[train_exp['month'] == 6]
                train_exp = train_exp.loc[train_exp['month'] != 6]
                
                train_w = train_exp['fraud_bool'].replace([0,1],[l,1])
                val_w = val_exp['fraud_bool'].replace([0,1],[l,1])

                print(len(train_w))
                train_x = train_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])
                val_x = val_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])

                train_y = (train_exp['decision'] == train_exp['fraud_bool']).astype(int)
                val_y = (val_exp['decision'] == val_exp['fraud_bool']).astype(int)

                if not (os.path.exists(f'./ova/{scen}-l_{l}/{train}/{expert}/')):
                    os.makedirs(f'./ova/{scen}-l_{l}/{train}/{expert}/')
                
                if not (os.path.exists(f'./ova/{scen}-l_{l}/{train}/{expert}/best_model.pickle')):
                    opt = hpo.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'./ova/{scen}-l_{l}/{train}/{expert}/')
                    opt.initialize_optimizer(CATEGORICAL_COLS, 10)

            

for scen in os.listdir('../data/alerts'):
    if len(scen.split('-')) == 3:
        sub = True
    else:
        sub = False
    for l in costs:
        scen = scen.split('.parquet')[0]
        if sub and (l not in cfg['run_sub']):
            continue
        test = pd.read_parquet(f'../testbed/testbed/{scen}-l_{l}/test/test.parquet')
        test_expert_pred = pd.read_parquet(f'../testbed/testbed/{scen}-l_{l}/test/test_expert_pred.parquet')

        expert_ids_path = f'../experts/teams/{scen}-l_{l}/expert_info/expert_ids.yaml'
        
        with open(expert_ids_path, 'r') as infile:
            EXPERT_IDS = yaml.safe_load(infile)

        cat_dict['assignment'] = EXPERT_IDS['human_ids'] + EXPERT_IDS['model_ids']

        test = cat_checker(test, data_cfg['data_cols']['categorical'], cat_dict)

        test = test.drop(columns = ['month','fraud_label'])
        preds = dict()

        for env in os.listdir(f'./ova/{scen}-l_{l}'):
            table = pd.DataFrame(index = test.index, columns = os.listdir(f'./ova/{scen}-l_{l}/{env}'))
            for expert in os.listdir(f'./ova/{scen}-l_{l}/{env}'):
                
                with open(f"./ova/{scen}-l_{l}/{env}/{expert}/best_model.pickle", "rb") as input_file:
                    model = pickle.load(input_file)
                
                table.loc[:, expert] = model.predict_proba(test)[:,1]
            
            for expert in EXPERT_IDS['model_ids']:
                table.loc[:,expert] = np.maximum(test_expert_pred[expert],  1-test_expert_pred[expert])
            
            preds[env.split('#')[0] + '#' + env.split('#')[1]] = table
        os.makedirs(f"../deferral/test_preds/{scen}-l_{l}/", exist_ok = True)
        with open(f"../deferral/test_preds/{scen}-l_{l}/ova.pkl", "wb") as out_file:
            pickle.dump(preds, out_file)


