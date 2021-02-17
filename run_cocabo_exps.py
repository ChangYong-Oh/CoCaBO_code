# -*- coding: utf-8 -*-
#==========================================
# Title:  run_cocabo_exps.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

# =============================================================================
#  CoCaBO Algorithms 
# =============================================================================
import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import socket
import testFunctions.syntheticFunctions
from methods.CoCaBO import CoCaBO
from methods.BatchCoCaBO import BatchCoCaBO


def CoCaBO_Exps(obj_func, budget, trials, kernel_mix, initN=10, batch=None):

    # define saving path for saving the results
    hostname = socket.gethostname()
    if hostname == 'DTA160000':
        saving_path = f'/home/coh1/Experiments/ContextualBO/{obj_func}_CoCaBO/'
    elif hostname[:4] == 'node':  # DAS5
        saving_path = f'/var/scratch/coh/Experiments/ContextualBO/{obj_func}_CoCaBO/'
    else:
        raise NotImplementedError
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # define the objective function
    if obj_func == 'Func2C':
        f = testFunctions.syntheticFunctions.func2C
        categories = [3, 5]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]

    elif obj_func == 'Func3C':
        f = testFunctions.syntheticFunctions.func3C
        categories = [3, 5, 4]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
            {'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]

    elif obj_func == 'Ackley5C':
        f = testFunctions.syntheticFunctions.ackley5C
        categories = [17, 17, 17, 17, 17]

        bounds = [
            {'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)},
            {'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)},
            {'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)},
            {'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)}]

    elif obj_func == 'SVMBoston':
        f = testFunctions.syntheticFunctions.svm_boston
        categories = [4, 2, 2]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
                  {'name': 'h2', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'h3', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-4, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-6, 0)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-6, 0)}]

    elif obj_func == 'XGBFashionMNIST':
        f = testFunctions.syntheticFunctions.xgboost_fashion_mnist
        categories = [10, 2, 2, 2]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)},
                  {'name': 'h2', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'h3', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'h4', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-6, 0)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-4, 1)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-3, 0)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (0, 5)}]

    else:
        raise NotImplementedError

    # Run CoCaBO Algorithm
    if batch == 1:
        # sequential CoCaBO
        mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds,
                       acq_type='LCB', C=categories,
                       kernel_mix = kernel_mix)

    else:
        # batch CoCaBO
        mabbo = BatchCoCaBO(objfn=f, initN=initN, bounds=bounds,
                            acq_type='LCB', C=categories,
                            kernel_mix=kernel_mix,
                            batch_size=batch)
    mabbo.runTrials(trials, budget, saving_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function',
                        type=str)
    parser.add_argument('-mix', '--kernel_mix',
                        help='Mixture weight for production and summation kernel. Default = 0.0',
                        type=float)
    parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
                        default=200, type=int)
    # parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 5',
    #                     default=5, type=int)
    parser.add_argument('-b', '--batch', help='Batch size (>1 for batch CoCaBO and =1 for sequential CoCaBO). Default = 1',
                        default=1, type=int)
    parser.add_argument('-s', '--seed', help='0,1,2,3,4', type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    obj_func = args.func
    kernel_mix = args.kernel_mix
    n_itrs = args.max_itr
    # n_trials = args.trials
    seed = args.seed
    batch = args.batch

    CoCaBO_Exps(obj_func=obj_func, budget=n_itrs, trials=range(seed, seed + 1), kernel_mix=kernel_mix, batch=batch)
