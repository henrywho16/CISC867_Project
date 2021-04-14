#
# SPDX-FileCopyrightText: 2020 Idiap Research Institute <contact@idiap.ch>
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#            Florian Mai <florian.mai@idiap.ch>
#            Thijs Vogels <thijs.vogels@epfl.ch>
#
# SPDX-License-Identifier: MIT
#

import os
import sys
import random
sys.path.append("./DeepOBS")
from deepobs import analyzer
from deepobs.analyzer import get_hyperparameter_optimization_performance, plot_hyperparam_optimization, print_tunability_to_latex, plot_box_hyperparam_optim
import random
import multiprocessing
random.seed(a=12) # fix random seed for when shuffling the random search results.
import pickle, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
from collections import defaultdict
import numpy as np

def main(Path,Problem,Optimizers,Num_shuffle):
    inpath = Path
    problems = Problem
    optimizers = Optimizers
    num_shuffle = Num_shuffle
    budget = 30
    outfile = Problem[0] +"_relative_performance"
    x_axis = "trials"

    # plot configuration
    #pgf_with_latex = {                      # setup matplotlib to use latex for output
    #    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    #    "text.usetex": args.usetex,                # use LaTeX to write all text
    #    "font.family": "serif",
    #    "font.serif": [],                   # blank entries should cause plots 
    #    "font.sans-serif": [],              # to inherit fonts from the document
    #    "font.monospace": [],
    #    "axes.labelsize": 10,
    #   "font.size": 10,
    #    "legend.fontsize": 8,               # Make the legend/label fonts 
    #    "xtick.labelsize": 14,               # a little smaller
    #    "ytick.labelsize": 12,                                    # default fig size of 0.9 textwidth
    #    "pgf.preamble": [
    #        r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts 
    #        r"\usepackage[T1]{fontenc}",        # plots will be generated 
    #        ]                                   # using this preamble
    #    }
    # mpl.use("pgf")
    #mpl.rcParams.update(pgf_with_latex)

    random.seed(a=12) # fix random seed for when shuffling the random search results.

    rankings = [i for i in range(1, budget)]
    x_axis = x_axis
    opt_labels  = optimizers
    def compute_problem_performance(p):
        problem = p
        prob_dir = inpath
        print(f"Doing {problem}")
        problem_type = ['acc', 'loss'][('vae' in problem) or ('deep' in problem)]

        dir_path = []
        for opt in optimizers:
            path_to_jsons = find_json_parent(os.path.join(prob_dir, opt))
            dir_path.append([path_to_jsons])
        opt_labels = optimizers
        hyperparam_perf = analyzer.compute_tunability_metrics(dir_path, opt_labels, obj="max" if problem_type =="acc" else "min", num_shuffle=num_shuffle, x_axis=x_axis, rankings=rankings)
        return hyperparam_perf

    def find_json_parent(rootdir):
        for subdir, dirs, files in os.walk(rootdir):
            for d in dirs:
                d = os.path.join(subdir, d)
                for f in os.listdir(d):
                    is_correct_dir = f.endswith(".json")
                    if is_correct_dir:
                        return subdir

    problem_performances = list(map(compute_problem_performance, problems))

    rank_perfs = {str(r) : {} for r in rankings}
    for rank in rankings:
        for l in opt_labels:
            perfs = []
            for p in problem_performances:
                #print(p[l][str(rank)])
                perfs.append(p[l]["r="+str(rank)])
                
            rank_perfs[str(rank)][l] = sum(perfs) / len(perfs)

    # print(dict_str)
    pgf_with_latex = {                      # setup mpl to use latex for output
                  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,               # Make the legend/label fonts
        "xtick.labelsize": 14,               # a little smaller
        # default fig size of 0.9 textwidth
        "ytick.labelsize": 12,
        # "axes.labelsize": 'large',
        # "legend.fontsize": 'large'
        # "axes.fontsize": 14
    }

    mpl.rcParams.update(pgf_with_latex)

    result_dict = rank_perfs
    print(result_dict)

    results_by_optimizer = defaultdict(list)
    vals = []
    for k,v in result_dict.items():
        vals.append(str(k))
        for k1,v1 in v.items():
            results_by_optimizer[k1].append(v1)

    #fig = plt.figure(figsize=(6.6*2, 4.95*2))
    fig = plt.figure()
    for o in opt_labels:
        # x = np.log2(np.array(vals, dtype = np.float32))
        x = np.array(vals).astype(np.float32)
        y = np.array(results_by_optimizer[o])

        sb.lineplot(x,y,label = o, sort = True)

    ax = fig.axes[0]
    plt.xlabel("# Hyperparameter Configurations (Budget)")
    plt.ylabel("Aggregated relative performance")
    # plt.xscale('log')
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(list(range(0, budget, 10)), 10))

    sb.despine(offset=0, trim=True)


    # plt.xticks(np.concatenate([x[:-1].astype(np.int32),x[-1].astype(np.float32)))
    plt.tight_layout(0)
    plt.savefig(outfile)
p1= os.path.abspath(os.getcwd()) + "\\output"
opt_list = ['adam','adamlr', 'sgdmcwc','sgdmcwclr']
problem_lst = ['fmnist_vae','imdb_bilstm','quadratic_deep'] 

main(p1,problem_lst,opt_list,100)
