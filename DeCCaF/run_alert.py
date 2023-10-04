import numpy as np
import pandas as pd
import pickle
import yaml
import os
from ortools.sat.python import cp_model
import random

baf_model_score = pd.read_parquet('./data/BAF_deployment_score.parquet')
baf = pd.read_parquet('./data/BAF.parquet')

with open('./ml_model/model/model_properties.pickle', 'rb') as infile:
        model_properties = pickle.load(infile)

l = model_properties['threshold']/(1-model_properties['threshold'])

orig_t = model_properties['threshold']

exp_pred = pd.read_parquet('./experts/expert_info/deployment_predictions.parquet').drop(columns = 'model#0')

test_index = baf.loc[baf['month'] == 7].index
val_index = baf.loc[baf['month'] == 6].index

val = baf_model_score.loc[val_index]
test = baf_model_score.loc[test_index]
train = baf_model_score.drop(val_index).drop(test_index)

data_cfg_path = './data/dataset_cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

l2a_fp_pred = pd.DataFrame(index = test.index, columns = exp_pred.columns )
l2a_fn_pred = pd.DataFrame(index = test.index, columns = exp_pred.columns )

Classes = np.array(['fn', 'fp', 'tn', 'tp'])

with open('./expertise_models/calibrated_predictions.pkl', 'rb') as fp:
        l2a_model_preds = pickle.load(fp)

with open('./ova_models/ova_predictions.pkl', 'rb') as fp:
        ova_model_preds = pickle.load(fp)


def rand_deferral_func(capacities, batches, testset, expert_preds, env, model):
    if os.path.isdir(f'./deferral_results/{model}/'  + env):
        print('done')
        return

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        human_cap = c.iloc[2:].sum()
        to_review = cases.sort_values(by = 'model_score', ascending = False)
        to_review = to_review.iloc[:human_cap,:]    
        experts = expert_preds.columns.to_list()
        for ix in to_review.index:
            done = 0
            while (done != 1):
                choice  = random.choice(experts)
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    done = 1
                else:
                    experts.remove(choice)
                if len(choice) == 0:
                    done = 1
            
    if not os.path.isdir(f'./deferral_results/{model}/'  + env):
        os.makedirs(f'./deferral_results/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'./deferral_results/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'./deferral_results/{model}/' + env +'/results.parquet')

    return assignments, results

def ova_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed):
    if os.path.isdir(f'./deferral_results/{seed}/{model}'  + env):
        print('done')
        return
    print(f'Calculating OvA for {env}')
    model_preds = model_preds[seed]
    seed = seed.split('#')[0] + '_' +  seed.split('#')[1]
    print(seed)
    
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        human_cap = c.iloc[2:].sum()
        to_review = cases.sort_values(by = 'model_score', ascending = False)
        to_review = to_review.iloc[:human_cap,:]
        preds = model_preds.loc[to_review.index]
        for ix, row in preds.iterrows():
            sorted = row.sort_values(ascending = False)
            for choice in sorted.index:
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    break

    if not os.path.isdir(f'./deferral_results/{seed}/{model}/' + env):
        os.makedirs(f'./deferral_results/{seed}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'./deferral_results/{seed}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'./deferral_results/{seed}/{model}/' + env +'/results.parquet')

    return assignments, results



def l2a_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed):
    if os.path.isdir(f'./deferral_results/{seed}/{model}/' + env):
        print('done')
        return 0

    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for i in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == i,:].iloc[0]
        cases = testset.loc[batches.loc[batches['batch'] == i]['case_id'],:]
        human_cap = c.iloc[2:].sum()
        to_review = cases.sort_values(by = 'model_score', ascending = False)
        to_review = to_review.iloc[:human_cap,:]    
        preds = model_preds.loc[to_review.index]
        for ix, row in preds.iterrows():
            sorted = row.sort_values()
            for choice in sorted.index:
                if c[choice]>0:
                    c[choice] -= 1
                    assignments.loc[ix] = choice
                    results.loc[ix] = expert_preds.loc[ix, choice]
                    break

    
    if not os.path.isdir(f'./deferral_results/{seed}/{model}/' + env):
        os.makedirs(f'./deferral_results/{seed}/{model}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'./deferral_results/{seed}/{model}/' + env +'/assignments.parquet')
    results.to_parquet(f'./deferral_results/{seed}/{model}/' + env +'/results.parquet')

    return assignments, results

def l2a_lin_deferral_func(capacities, batches, testset, expert_preds, model_preds, env, model, seed):
    if os.path.isdir(f'./deferral_results/{seed}/{model}/'  + env):
        print('done')
        return

    model_name = model
    assignments = pd.DataFrame(columns = ['assignment'])
    results = pd.DataFrame(columns = ['prediction'])
    for b in np.arange(1,batches['batch'].max()+1):
        c = capacities.loc[capacities['batch_id'] == b,:].iloc[0]
        cases = testset.loc[batches.loc[batches['batch'] == b]['case_id'],:]
        human_cap = c.iloc[2:].sum()
        to_review = cases.sort_values(by = 'model_score', ascending = False)
        to_review = to_review.iloc[:human_cap,:]    
        preds = model_preds.loc[to_review.index]
        cost_matrix_df = preds.T
        for d in c.index:
            if c.loc[d] == 0:
                cost_matrix_df = cost_matrix_df.drop(index=d)
                

        cost_matrix = cost_matrix_df.values
        num_workers, num_tasks = cost_matrix.shape
        workers = list(cost_matrix_df.index)

        model = cp_model.CpModel()
        x = []
        for i in range(num_workers):
            t = []
            for j in range(num_tasks):
                t.append(model.NewBoolVar(f'x[{i},{j}]'))
            x.append(t)

        # capacity constraints
        for i in range(num_workers):
            model.Add(sum([x[i][j] for j in range(num_tasks)]) == c[workers[i]])

        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            model.AddExactlyOne(x[i][j] for i in range(num_workers))

        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(cost_matrix[i, j] * x[i][j])
        model.Minimize(sum(objective_terms))
        #This sum(objective_terms) is the loss of the batch.
        solver = cp_model.CpSolver()
        # solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if not status == cp_model.OPTIMAL and not status == cp_model.FEASIBLE:
            print('Solution not found!')
            return None

        print('Batch solved')
        
        for j in range(num_tasks):
            ix = cost_matrix_df.columns.to_list()[j]
            for i in range(num_workers):
                if solver.BooleanValue(x[i][j]):
                    assignments.loc[ix] = workers[i]
                    results.loc[ix] = expert_preds.loc[ix, workers[i]]

    
    
    if not os.path.isdir(f'./deferral_results/{seed}/{model_name}/'  + env):
        os.makedirs(f'./deferral_results/{seed}/{model_name}/' + env)

    assignments = assignments.astype('object')
    assignments.to_parquet(f'./deferral_results/{seed}/{model_name}/' + env +'/assignments.parquet')
    results.to_parquet(f'./deferral_results/{seed}/{model_name}/' + env +'/results.parquet')

    return assignments, results

a = dict()
for direc in os.listdir('./testbed/test'):
    print('./testbed/test/' + direc)
    if os.path.isfile('./testbed/test/' + direc):
        continue
    a[direc] = dict()
    a[direc]['bat'] = pd.read_csv('./testbed/test/' + direc + '/batches.csv')
    a[direc]['cap'] = pd.read_csv('./testbed/test/' + direc + '/capacity.csv')

# %%
from joblib import Parallel, delayed

Parallel(n_jobs=25)(
            delayed(rand_deferral_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                env,
                'random'
            )
            for env in a 
        )

for seed in l2a_model_preds:

    for expert in exp_pred.columns.to_list():
        l2a_fp_pred[expert] = l2a_model_preds[seed][expert]['fp']
        l2a_fn_pred[expert] = l2a_model_preds[seed][expert]['fn']

    
    costs = l*l2a_fp_pred + l2a_fn_pred
    
    Parallel(n_jobs=25)(
            delayed(l2a_deferral_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                l2a_model_preds,
                env,
                'DeCCaF_greedy',
                seed
            )
            for env in a 
        )
    
    Parallel(n_jobs=25)(
            delayed(l2a_lin_deferral_func)(
                a[env]['cap'],
                a[env]['bat'],
                test,
                exp_pred,
                l2a_model_preds,
                env,
                'DeCCaF_linear',
                seed
            )
            for env in a 
        )
    
    
    
    





