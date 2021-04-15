# CISC867 Reproducibility Project: Optimizer Benchmarking Needs to Account for Hyperparameter Tuning 

The source code for the paper can be found in this https://github.com/idiap/hypaobp

My code is entirely based on the author's source code. The original project has a specific set of operating enviorment. During the enviorment setup process, the original environment setup file: spec-file.txt is not working. Seems like packages with .conda extension are repalced with .tar.bz2 in the repo.anaconda.com. 
Use my spec-file-update.txt to create an environment

After setting up the environment. 

Put my fast_runner to the root folder, it should work!

To analiyze the outcome, put fast_compute_realtive_performances.py to the root folder. It generate graph with a single click.

output.7z is contain my experiment comeout.

analyze_tunability is used to generate figure 1 and figure 3 for my report.
Unfortunantaly, I had difficulty automate analyze_tunability script. Seems like multi-process does not work well with my modfication :(
Also, analyze_tunability script needs a PDF generation tool such as miktex.

To generate Stacked Probability Plot as figure 6:

python results_create_dataframe.py -inpath <Input file path> 
python results_process.py -inpath <root path> -budget 30 
python results_analyze_winprob.py -inpath <root path> -optimizers adam adamlr sgdmcwc sgdmcwclr -budget 30  -global_averaging
