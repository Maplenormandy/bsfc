# bsfc
Bayesian Spectral Fitting Code
[Github page](https://github.com/Maplenormandy/bsfc)

Under active development. 
N.M.Cao & F.Sciortino
 
 
BSFC is a Python code that decomposes atomic spectra into a set of 3 (or more) Hermite polynomials per line, with central wavelength and width scaling optimized as joint parameters for all lines. MCMC-based algorithms (some interfacing Fortran) are used to obtain Bayesian estimates of the moments of each line. The first 3 of these are required to estimate line intensity, particle velocity and temperature. 


Installation
-------------------
Installing BSFC is relatively simple. Use

 pip install bsfc

However you might want to git clone this repo to get the latest version of the code. 

Note at this time all package dependencies must be handled individually by the user. Other than numpy, scipy and other Python basics, the following packages must be installed:

* emcee: see http://dfm.io/emcee/current/. Installing this is as simple as git-cloning it from https://github.com/dfm/emcee. 

* MultiNest: this is strictly only necessary if nested sampling is used, but it is required at this time. You will need to git-clone the Python wrapper PyMultiNest (https://github.com/JohannesBuchner/PyMultiNest/tree/master/pymultinest) and follow instructions on this page to build MultiNest itself on your system. 

* The Python multiprocessing package and mpi4py are also required for parallelization. Standard package managers such as pip should do all the work for you. However, to use parallelization in MultiNest (necessary to achieve reasonable speed), MPI must also be installed. We successfully tested the framework with both OpenMPI and IMPI. Note that PyMultiNest has issues with Intel compilers, but these can be solved with some simple hacks (get in touch with us if you need help!). 

Running tips
------------

For users on the MGHPCC engaging cluster, you may use installations already set up on the system. You can do 

 module use /home/sciortino/modulefiles
 
 module load sciortino/bsfc
To obtain an interactive allocation, use 

 srun -N 1 -n 32 --exclusive -p <NAME-OF-PARTITION> -time 00:29:00 
for 32 CPUs, for example. The code must be run from the directory where this repository is git-cloned. 

The code is parallelized on multiple levels. To obtain detailed statistics for a specific time slice, it is convenient to use 

 python bsfc_run.py <SHOT> <OPTION> <NSTEPS>

where <SHOT> is the tokamak discharge number (currently, only Alcator C-Mod data loading is enabled), <OPTION> specifies the kind of run to be done (set to 1 for a single-spectrum inference) and <NSTEPS> is the number of steps to be used for MCMC algorithms. Note that nested sampling, implemented via MultiNest, does not require specification of a number of steps (it runs until a pre-set convergence condition). To view the results of spectral fitting, simply repeat the same command as above and some example plots will be displayed. 
 
To obtain spectral analysis for an entire tokamak discharge time-interval using MultiNest, across all Hirex-Sr channels, use
 
 mpirun python bsfc_run_mpi.py <SHOT> 
 
This code offers a more up-to-date version of nested-sampling-based routines. 
 
To view the results after running the mpirun command, re-run the same command without mpirun, i.e. use
 
 python bsfc_run_mpi.py <SHOT> 
