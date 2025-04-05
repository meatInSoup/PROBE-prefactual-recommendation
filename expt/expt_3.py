import os
import numpy as np
import scipy
import pandas as pd
import joblib
import torch
import sklearn
import copy
from collections import defaultdict, namedtuple
from joblib import parallel_backend
from joblib.externals.loky import set_loky_pickler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

import dice_ml

from utils import helpers
from utils.transformer import get_transformer
from utils.data_transformer import DataTransformer
from utils.funcs import compute_max_distance, lp_dist

from expt.common_l1 import synthetic_params, synthetic_params_mean_cov, clf_map, method_map
from expt.common_l1 import _run_single_instance, to_mean_std
from expt.common_l1 import load_models, enrich_training_data
from expt.common_l1 import method_name_map, dataset_name_map, metric_order, metric_order_graph
from expt.expt_config import Expt1, Expt3


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger):
    print("Running dataset: %s, classifier: %s, method: %s..."
               % (dname, cname, mname))
    full_l = ["synthesis", "german"]
    df, numerical = helpers.get_dataset(dname, params=synthetic_params) if dname in full_l else helpers.get_full_dataset(dname, params=synthetic_params)
    full_dice_data = dice_ml.Data(dataframe=df,
                     continuous_features=numerical,
                     outcome_name='label')
    transformer_dice = DataTransformer(full_dice_data)
    
    # transform data for classifier
    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    X_dice = transformer_dice.transform(X_df).to_numpy()
    
    # transform data for recourse
    transformer = get_transformer(dname)
    X = transformer.transform(df)
    print(X_dice.shape[1])
    print(X.shape[1])
    
    cat_indices = [len(numerical) + i for i in range(X.shape[1] - len(numerical))]

    X_train_dice, X_test_dice, y_train, y_test = train_test_split(X_dice, y, train_size=0.8, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    if mname == 'reup_graph' and dname == 'gmc':
        np.random.seed(seed)
        idx_train = np.random.choice(len(X_train_dice), 40000, replace=False)
        X_train_dice = X_train_dice[idx_train]
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

    
    new_config = Expt3(ec.to_dict())
    
    d = X.shape[1]
    clf = clf_map[cname]
    model = load_models(dname, cname, wdir)
    method = method_map[mname]

    l1_cost = []
    valid = []
    rank = []
    feasible = []
    log = []

    y_pred = model.predict(X_test_dice)
    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]
    # np.random.seed(seed)
    # indices = np.random.choice(uds_X.shape[0], size=ec.max_ins, replace=False)
    # uds_X, uds_y = uds_X[indices], uds_y[indices]
    
    # all_A = np.zeros((ec.num_A, X_train.shape[1], X_train.shape[1]))
    all_w = np.zeros((ec.num_w, X_train.shape[1], 1))
    idx = 0
    d = np.zeros(X_train.shape[1])
    d[0] = 1.0
    np.random.seed(seed)
    for i in range(ec.num_w):
        shape = X_train.shape[1]
        np.random.shuffle(d)
        w = np.random.rand(shape, 1)
        w = w / np.sum(w)
        all_w[i] = w
        
    predict_label = model.predict(X_train_dice)

    params = dict(train_data=X_train,
                  labels=model.predict(X_train_dice),
                  dataframe=df,
                  numerical=numerical,
                  config=new_config,
                  method_name=mname,
                  dataset_name=dname,
                  transformer=transformer,
                  cat_indices=cat_indices,
                  A=None,
                  num_w=ec.num_w,
                  w=w,
                  all_w=all_w,)

    params['reup_params'] = ec.reup_params
    params['bayepr_params'] = ec.bayepr_params
    params['bayepr_linearized_params'] = ec.bayepr_linearized_params
    params['bayepr_diag_params'] = ec.bayepr_diag_params
    params['wachter_params'] = ec.wachter_params
    params['dice_params'] = ec.dice_params
    params['face_params'] = ec.face_params
    params['cost_type'] = "weighted-l1"

    jobs_args = []

    for idx, x0 in enumerate(uds_X):
        jobs_args.append((idx, method, x0, model, seed, logger, params))

    rets = joblib.Parallel(n_jobs=min(num_proc, 32), backend='loky')(joblib.delayed(_run_single_instance)(*jobs_args[i]) for i in range(len(jobs_args)))

    for ret in rets:
        l1_cost.append(ret.l1_cost)
        valid.append(ret.valid)
        rank.append(ret.rank)
        feasible.append(ret.feasible)
        log.append(ret.log_dict)

    def to_numpy_array(lst):
        pad = len(max(lst, key=len))
        return np.array([i + [0]*(pad-len(i)) for i in lst])

    l1_cost = np.array(l1_cost)
    valid = np.array(valid)
    rank = np.array(rank)
    feasible = np.array(feasible)

    res = {}
    res['ptv_name'] = "T"
    res['ptv_list'] = np.array([i for i in range(params["reup_params"]["T"])])
    res['rank'] = rank
    
    log_exp = {}
    log_exp['log'] = log
    log_exp['data'] = X_train
    log_exp['label'] = predict_label

    helpers.pdump((l1_cost, valid, feasible),
                  f'{cname}_{dname}_{mname}_expt1.pickle', wdir)
    helpers.pdump(res, f'{cname}_{dname}_{mname}_expt2.pickle', wdir)
    helpers.pdump(log_exp, f'{cname}_{dname}_{mname}_expt3.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)

def plot_1(ec, wdir, cname, datasets, methods):
    res = defaultdict(list)
    res2 = defaultdict(list)

    for mname in methods:
        res2['method'].append(method_name_map[mname])

    for i, dname in enumerate(datasets):
        res['dataset'].extend([dataset_name_map[dname]] + [""] * (len(methods) - 1))

        joint_feasible = None
        for mname in methods:
            _, _, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}_expt1.pickle', wdir)
            if joint_feasible is None:
                joint_feasible = np.ones_like(feasible)
            if '_ar' in mname:
                joint_feasible = np.logical_and(joint_feasible, feasible)

        temp = defaultdict(dict)

        for metric, order in metric_order.items():
            temp[metric]['best'] = -np.inf

        for mname in methods:
            l1_cost, valid, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}_expt1.pickle', wdir)
            avg = {}
            avg['cost'] = l1_cost 
            avg['valid'] = valid 

            for metric, order in metric_order.items():
                m, s = np.mean(avg[metric]), np.std(avg[metric])
                temp[metric][mname] = (m, s)
                temp[metric]['best'] = max(temp[metric]['best'], m * order)

            temp['feasible'][mname] = np.mean(feasible)

        for mname in methods:
            res['method'].append(method_name_map[mname])
            for metric, order in metric_order.items():
                m, s = temp[metric][mname]
                is_best = (temp[metric]['best'] == m * order)
                res[metric].append(to_mean_std(m, s, is_best))
                res2[f"{metric}-{dname[:2]}"].append(to_mean_std(m, s, is_best))

            res[f'feasible'].append("{:.2f}".format(temp['feasible'][mname]))

    df = pd.DataFrame(res)
    filepath = os.path.join(wdir, f"{cname}{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')

    df = pd.DataFrame(res2)
    filepath = os.path.join(wdir, f"{cname}_hor{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')


def run_expt_3(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None, rerun=True):
    logger.info("Running ept 1...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e3.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e3.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e3.all_methods

    jobs_args = []
    if not plot_only:
        for cname in classifiers:
            cmethods = copy.deepcopy(methods)
            if cname == 'rf' and 'wachter' in cmethods:
                cmethods.remove('wachter')            

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}_expt1.pickle")
                    if (not os.path.exists(filepath)) or rerun:
                        jobs_args.append((ec.e3, wdir, dname, cname, mname,
                            num_proc, seed, logger))

        # rets = joblib.Parallel(n_jobs=4, backend='loky')(joblib.delayed(run)(
        #     *jobs_args[i]) for i in range(len(jobs_args)))
        ret = run(*jobs_args[0])

    # for cname in classifiers:
    #     cmethods = copy.deepcopy(methods)
    #     if cname == 'rf' and 'wachter' in cmethods:
    #         cmethods.remove('wachter')            

    #     plot_1(ec.e3, wdir, cname, datasets, cmethods)

    logger.info("Done ept 3.")
