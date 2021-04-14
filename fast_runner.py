#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#

import argparse
from functools import partial, update_wrapper

import torch
from scipy.stats.distributions import binom, uniform

import sys
sys.path.append("./DeepOBS/")
import deepobs
from deepobs import config
from deepobs.tuner import RandomSearch
from deepobs.tuner.sampler import log_uniform, negative_log_uniform, tuple_sampler, log_normal, negative_log_normal
config.set_framework('pytorch')

if config.get_framework() == 'tensorflow':
    raise ValueError("Only is supported.")

else:
    
    from deepobs.pytorch.runners import StandardRunner, LearningRateScheduleRunner
    deepobs.pytorch.config.set_default_device('cuda:0')
    OPTIM_MAP = {
        'adadelta': torch.optim.Adadelta,
        'adam': torch.optim.Adam,
        'adamlr': torch.optim.Adam,
        'adamlreps': torch.optim.Adam,
        'adamwclr':torch.optim.Adam,
        'adamwlr':torch.optim.Adam,
        'adagrad': torch.optim.Adagrad,
        'sgd': torch.optim.SGD,
        'sgdm': torch.optim.SGD,
        'sgdmc': torch.optim.SGD,
        'sgdmcwc': torch.optim.SGD,
        'sgdmw': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop, 
        'sgdmcwclr': deepobs.pytorch.optimizers.SGDDecay,
        'adamwclrdecay': deepobs.pytorch.optimizers.AdamDecay
    }

    OPTIM_PARAMETERS = {
        'adadelta': {"lr": {'type': float}, "rho": {'type': float}, "eps": {'type': float}},
        'adam': {"lr": {'type': float}, "betas": {"beta1": {'type': float}, "beta2": {'type': float}}, "eps": {'type': float}},
        'adamlr': {"lr": {'type': float}},
        'adamlreps': {"lr" : {'type': float}, "eps": {'type': float}},
        'adamwclr': {"lr" : {'type': float}, "weight_decay" : {'type' : float}},
        'adamwlr': {"lr" : {'type': float}, "weight_decay" : {'type' : float}},
        'adagrad': {"lr": {'type': float}},
        'sgd': {"lr": {'type': float}},
        'sgdm': {"lr": {"type": float}, "momentum": {"type": float}},
        'sgdmc': {"lr": {"type": float}, "momentum": {"type": float}},
        'sgdmcwc': {"lr": {"type": float}, "momentum": {"type": float}, "weight_decay":{"type": float}},
        'sgdmw': {"lr": {"type": float}, "momentum": {"type": float}, "weight_decay":{"type": float}},
        'sgdm': {"lr": {"type": float}, "momentum": {"type": float}},
        'rmsprop': {"lr": {"type": float}, "eps": {'type': float}},
        'sgdmcwclr': {"lr": {"type": float}, "momentum": {"type": float}, "weight_decay":{"type": float}, "power": {"type": float}},
        'adamwclrdecay': {"lr": {"type": float}, "weight_decay":{"type": float}, "power": {"type": float}}
    }
        
    OPTIM_SAMPLERS_TUNED_PRIOR = {
        'sgd' : {'lr' : log_normal(-2.0957990019132593, 1.3121766191485007)},
        'adam' : {'lr' : log_normal(-2.6967872556688617, 1.4208894762774538), 'betas': tuple_sampler(negative_log_uniform(-5, -1), negative_log_uniform(-5, -1)),
                 'eps': log_uniform(-8, 0)},
        'adamlr' : {'lr' : log_normal(-3.0212300204337854, 1.2708856424869013)},
        'adamlreps' : {'lr' : log_normal(-3.0212300204337854, 1.2708856424869013), 'eps': log_uniform(-10,10)},
        'adamwclr' : {'lr' : log_normal(-3.0212300204337854, 1.2708856424869013), 'weight_decay' : uniform(10 ** -5, 0)},
        'adamwlr' : {'lr' : log_normal(-3.0212300204337854, 1.2708856424869013), 'weight_decay' : log_uniform(-5, -1)},
        'adagrad' : {'lr' : log_normal(-2.004601189249433, 1.204469231243577)},
        'sgdm' : {'lr' : log_normal(-2.3789883206086384, 1.4130924408241188), 'momentum' : uniform(0, 1)},
        'sgdmc' : {'lr' : log_normal(-2.342230523721803, 1.5525436071081387), 'momentum': uniform(0.9, 0)},
        'sgdmcwc' : {'lr' : log_normal(-2.4543935562228762, 1.3641391791655386), 'momentum': uniform(0.9, 0), 'weight_decay' : uniform(10 ** -5, 0)},
        'sgdmw' : {'lr' : log_normal(-2.438163785483538, 1.2040263166410041), 'momentum' : uniform(0, 1), 'weight_decay' : log_uniform(-5, -1)},
        'sgdmcwclr': {'lr' : log_normal(-2.4543935562228762, 1.3641391791655386), 'momentum': uniform(0.9, 0), 'weight_decay' : uniform(10 ** -5, 0), "power": uniform(0.5, 5)},
        'adamwclrdecay': {'lr' : log_normal(-3.0212300204337854, 1.2708856424869013), 'weight_decay' : uniform(10 ** -5, 0), "power": uniform(0.5, 5)}
    }
    PROBLEMS = ['cifar100_3c3d',
                'cifar100_allcnnc',
                'cifar10_3c3d',
                'fmnist_2c2d',
                'fmnist_mlp',
                'fmnist_vae',
                'mnist_2c2d',
                'mnist_logreg',
                'mnist_mlp',
                'mnist_vae',
                'quadratic_deep',
                'svhn_wrn164',
                'tolstoi_char_rnn',
                'imdb_bilstm'
                ]





def main(early_stopping,optimizer_class_o,problem_o,num_evals_o,random_seed_o,path_o):
    # config.set_framework('pytorch')
    config.set_early_stopping(early_stopping)
    optimizer_class = OPTIM_MAP[optimizer_class_o]
    hyperparams = OPTIM_PARAMETERS[optimizer_class_o]
    sampler = OPTIM_SAMPLERS_TUNED_PRIOR[optimizer_class_o]
    runner = StandardRunner

    if optimizer_class_o in ['sgdmcwclr', 'adamwclrdecay']:
        optimizer_class.set_max_epochs(config.get_testproblem_default_setting(problem_o)['num_epochs'])
        runner = LearningRateScheduleRunner
    tuner = RandomSearch(optimizer_class, hyperparams, sampler, runner=runner, ressources=num_evals_o)
    tuner.tune(problem_o, rerun_best_setting=False, output_dir=path_o, random_seed=random_seed_o, weight_decay=0)
optims = ['adam','adamlr','adagrad','adamwclrdecay','sgd','sgdm','sgdmc','sgdmcwc','sgdmw','sgdmcwclr'] 
##,'adamlr','adagrad','adamwclrdecay','sgd','sgdm','sgdmc','sgdmcwc','sgdmw','sgdmcwclr'
##'cifar100_allcnnc','cifar10_3c3d','fmnist_2c2d','fmnist_vae', 
##      'mnist_vae','quadratic_deep','svhn_wrn164','tolstoi_char_rnn','imdb_bilstm'


problem_list = ['cifar100_allcnnc','cifar10_3c3d','fmnist_2c2d',
               
                ['fmnist_vae', 'mnist_vae','quadratic_deep','imdb_bilstm']     ,'svhn_wrn164','tolstoi_char_rnn',]
for problem in problem_list:
    for i in optims:

        main(False,i,problem,1,1337,r"output/"+i)
