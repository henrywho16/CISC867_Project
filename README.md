# CISC867 Reproducibility Project: Optimizer Benchmarking Needs to Account for Hyperparameter Tuning 

The source code for the paper can be found in this https://github.com/idiap/hypaobp
 
My code is entirely based on the author's source code. This github repositoy acts as a suppliment for the orginal source code.

1. Clone the original soruce code.

2. Setup the environment.

The original project has a specific set of operating enviorment. During the enviorment setup process, the original environment setup file: spec-file.txt is not working. Packages with .conda extension are repalced with .tar.bz2 in the repo.anaconda.com. 
Instead use my spec-file-update.txt to create an environment
After setting up the environment, all of the required packages for model training are installed.

3. Model training.
Put my fast_runner to the root folder, double click and wait for 400+ hours.
output.7z contains the result from my training.

4.Analize outcome. 

Relative Performance (Figure 4). Put fast_compute_realtive_performances.py to the root folder. It will generate relative performance for all optimizers in one click.

analyze_tunability:
analyze_tunability is used to generate figure 1 and figure 3 for my report. It requires PDF generation tool such as miktex to work.
Unfortunantaly, I had difficulty automate analyze_tunability script. Seems like multi-process does not work well with my modfication :(

Here is the command to generate box plot for it:
<problem> : fmnist_vae, quadratic_deep,imdb_bilstm
python analyze_tunability.py -inpath <root path>-optimizers adam adamlr sgdmcwc sgdmcwclr -problem  <problem>  -outfile <problem>  -num_shuffle 100 
add -print_metrics at end to generate performance metric.
python analyze_tunability.py -inpath <root path>-optimizers adam adamlr sgdmcwc sgdmcwclr -problem  <problem>  -outfile <problem>  -num_shuffle 100 -print_metrics

To generate Stacked Probability Plot as figure 6:

python results_create_dataframe.py -inpath <Input file path> 
python results_process.py -inpath <root path> -budget 30 
python results_analyze_winprob.py -inpath <root path> -optimizers adam adamlr sgdmcwc sgdmcwclr -budget 30  -global_averaging
