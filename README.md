# CISC867 Reproducibility Project: Optimizer Benchmarking Needs to Account for Hyperparameter Tuning 

The source code for the paper can be found in this https://github.com/idiap/hypaobp

My code is entirely based on the author's source code. This github repository acts as a supplement for the original source code.

1. Clone the original source code.

2. Setup the environment.

3. The original project has a specific set of operating environment. During the environment setup process, the original environment setup file: spec-file.txt is not working. Packages with .conda extension are replaced with .tar.bz2 in the repo.anaconda.com. Instead use my spec-file-update.txt to create an environment After setting up the environment, all of the required packages for model training are installed.

Model training. Put my fast_runner to the root folder, double click and wait for 400+ hours. output.7z contains the result from my training.
4. Analyze outcome.

Relative Performance (Figure 4). Put fast_compute_relative_performances.py to the root folder. It will generate relative performance for all optimizers in one click.

analyze_tunability: analyze_tunability is used to generate figure 1 and figure 3 for my report. It requires PDF generation tool such as miktex to work. Unfortunately, I had difficulty automate analyze_tunability script. Seems like multi-process does not work well with my modification :(

Here is the command to generate box plot for it: 

<problem>: fmnist_vae, quadratic_deep,imdb_bilstm 
 python analyze_tunability.py -inpath -optimizers adam adamlr sgdmcwc sgdmcwclr -problem <problem> -outfile -num_shuffle 100 add -print_metrics at end to generate performance metric. 
  
 python analyze_tunability.py -inpath -optimizers adam adamlr sgdmcwc sgdmcwclr -problem <problem> -outfile -num_shuffle 100 -print_metrics

To generate Stacked Probability Plot as figure 6:

python results_create_dataframe.py -inpath python results_process.py -inpath -budget 30 python results_analyze_winprob.py -inpath -optimizers adam adamlr sgdmcwc sgdmcwclr -budget 30 -global_averaging
