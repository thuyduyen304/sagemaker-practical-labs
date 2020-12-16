import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb

import boto3

def _xgb_train(
        params,
        dtrain,
        evals,
        num_boost_round,
        num_fold,
        seed,
        model_dir,
        output_data_dir,
        is_master,
        early_stopping_round):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training, or is running single node training job. Note that rabit_run will include this argument.
    """
    # booster = xgb.train(
    # params=params,
    # dtrain=dtrain,
    # evals=evals,
    # num_boost_round=num_boost_round,
    # early_stopping_rounds=early_stopping_round)

    cvresult = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=num_fold,
        metrics=['auc'],
        early_stopping_rounds=early_stopping_round,
        stratified=True,
        seed=seed)

    if is_master:
        print("hello there")
        model_location = output_data_dir + '/cv-result'
        pkl.dump(cvresult, open(model_location, 'wb'))
        logging.info("Stored cv result at {}".format(model_location))

       # if is_master:
        # model_location = model_dir + '/xgboost-model'
        # pkl.dump(booster, open(model_location, 'wb'))
        # logging.info("Stored trained model at {}".format(model_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # cross validation params
    parser.add_argument('--k_fold', type=int)

    # Hyperparameters are described here. In this simple example we are just
    # including one hyperparameter.
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--early_stopping_round', type=int)
    parser.add_argument('--eval_metric', type=str)
    parser.add_argument('--num_fold', type=int)
    parser.add_argument('--seed', type=int)

    # Sagemaker specific arguments. Defaults are set in the environment
    # variables.
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
#     parser.add_argument('--validation', type=str,
#                         default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--sm_hosts', type=str, default=os.environ['SM_HOSTS'])
    parser.add_argument('--sm_current_host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(os.environ['SM_HOSTS'])
    sm_current_host = args.sm_current_host
    
    print("hello, i get data")

    dtrain = get_dmatrix(args.train, 'csv')
#     dval = get_dmatrix(args.validation, 'csv')
#     watchlist = [(dtrain, 'train'), (dval, 'validation')
#                  ] if dval is not None else [(dtrain, 'train')]
    watchlist = [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbose': args.verbose,
        'objective': args.objective,
        'eval_metric': args.eval_metric
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        num_fold=args.num_fold,
        seed=args.seed,
        model_dir=args.model_dir,
        early_stopping_round=args.early_stopping_round,
        output_data_dir=args.output_data_dir)

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


# def model_fn(model_dir):
#     """Deserialized and return fitted model.

#     Note that this should have the same name as the serialized model in the _xgb_train method
#     """
#     model_file = 'xgboost-model'
#     booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
#     return booster
