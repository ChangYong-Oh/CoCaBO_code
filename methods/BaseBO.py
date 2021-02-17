# -*- coding: utf-8 -*-
#==========================================
# Title:  BaseBO.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

import os
import pickle
import random

import numpy as np

MAX_RANDOM_SEED = 2 ** 31 - 1


class BaseBO():
    """
    Base class with common operations for BO with continuous and categorical
    inputs
    """

    def __init__(self, objfn, initN, bounds, C, rand_seed=108, debug=False,
                 batch_size=1, **kwargs):
        self.f = objfn  # function to optimise
        self.bounds = bounds  # function bounds
        self.batch_size = batch_size
        self.C = C  # no of categories
        self.initN = initN  # no: of initial points
        self.nDim = len(self.bounds)  # dimension
        self.rand_seed = rand_seed
        self.debug = debug
        self.saving_path = None
        self.kwargs = kwargs
        self.x_bounds = np.vstack([d['domain'] for d in self.bounds
                                   if d['type'] == 'continuous'])

    def initialise(self, seed):
        """Get NxN intial points"""
        data = []
        result = []

        print(f"Creating init data for seed {seed}")

        initial_data_x = np.zeros((self.initN, self.nDim))
        seed_list = np.random.RandomState(seed).randint(0, MAX_RANDOM_SEED - 1, self.nDim)

        n_discrete = len(self.C)
        n_continuous = self.nDim - n_discrete
        for d in range(n_continuous):
            low, high = self.bounds[n_discrete + d]['domain']
            initial_data_x[:, d] = np.random.RandomState(seed_list[d]).uniform(low, high, self.initN)
        for d in range(n_discrete):
            domain = self.bounds[d]['domain']
            initial_data_x[:, d + n_continuous] = \
                np.random.RandomState(seed_list[d + n_continuous]).randint(0, len(domain), self.initN)

        Zinit = np.hstack([initial_data_x[:, n_continuous:], initial_data_x[:, :n_continuous]]).astype(np.float32)
        yinit = np.zeros([initial_data_x.shape[0], 1])

        for j in range(self.initN):
            ht_list = list(Zinit[j, :n_discrete])
            yinit[j] = self.f(ht_list, Zinit[j, n_discrete:])
            # print(ht_list, Xinit[j], yinit[j])

        init_data = {}
        init_data['Z_init'] = Zinit
        init_data['y_init'] = yinit

        data.append(Zinit)
        result.append(yinit)
        return data, result

    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array(
                [np.random.uniform(bounds[b]['domain'][0],
                                   bounds[b]['domain'][1], 1)[0]
                 for b in range(nDim)])
        return Xinit

    def my_func(self, Z):
        Z = np.atleast_2d(Z)
        if len(Z) == 1:
            X = Z[0, len(self.C):]
            ht_list = list(Z[0, :len(self.C)])
            return self.f(ht_list, X)
        else:
            f_vals = np.zeros(len(Z))
            for ii in range(len(Z)):
                X = Z[ii, len(self.C):]
                ht_list = list(Z[ii, :len(self.C)].astype(int))
                f_vals[ii] = self.f(ht_list, X)
            return f_vals

    def save_progress_to_disk(self, *args):
        raise NotImplementedError

    def runTrials(self, trials, budget, saving_path):
        raise NotImplementedError
