# bsfc
Bayesian Spectral Fitting Code(s)

Under active development. 
N.Cao & F.Sciortino
 
 -----------------------------------------
 
BSFC is a Python code that decomposes atomic spectra into a set of 3 Hermite polynomials per line, with central wavelength and width scaling optimized as joint parameters for all lines. MCMC-based algorithms (some interfacing Fortran) are used to obtain Bayesian estimates of the 0th, 1st and 2nd moments of each line, corresponding to measurements of line intensity, particle velocity and temperature. 

Installing BSFC is very simple, but at this time all package dependencies must be handled individually by the user. Other than numpy, scipy and other Python basics (whose versions are currently from Python 2.7), the following packages must be installed:
- emcee: see http://dfm.io/emcee/current/ . Installing this is as simple as git-cloning it from https://github.com/dfm/emcee. 
- MultiNest: this is strictly only necessary if nested sampling is used, but it is required at this time. You will need to git-clone the Python wrapper PyMultiNest (https://github.com/JohannesBuchner/PyMultiNest/tree/master/pymultinest) and follow instructions on this page to build MultiNest itself on your system. 
- The Python multiprocessing package and mpi4py are also required for parallelization. Standard package managers such as pip should do all the work for you. However, to use parallelization in MultiNest (necessary to achieve reasonable speed), MPI must also be installed. We successfully tested the framework with both OpenMPI and IMPI. Note that PyMultiNest has issues with Intel compilers, but these can be solved with some simple hacks (get in touch with us if you need help!). 

 -----------------------------------------
 
For users on the engaging cluster in Massachusetts, you may use installations already set up on the system. You can do 

$ module use /home/sciortino/modulefiles
$ module load sciortino/bsfc

To obtain an interactive allocation, use 

$ salloc -N 1 -n 32 -p <NAME-OF-PARTITION> -time 00:29:00 

for 32 CPUs, for example. The code must be run from the directory where this repository is git-cloned. 

The code is parallelized on multiple levels. To obtain detailed statistics for a specific time slice, it is convenient to use 

$ python bsfc_run.py <SHOT> <OPTION> <NSTEPS>

where <SHOT> is the tokamak discharge number (currently, only Alcator C-Mod data loading is enabled), <OPTION> specifies the kind of run to be done (set to 1 for a single-spectrum inference) and <NSTEPS> is the number of steps to be used for MCMC algorithms. Note that nested sampling, implemented via MultiNest, does not require specification of a number of steps (it runs until a pre-set convergence condition). To view the results of spectral fitting, simply repeat the same command as above and some example plots will be displayed. 
 
 To obtain spectral analysis for an entire tokamak discharge time-interval, across all Hirex-Sr channels, use
 
 $ mpirun python bsfc_run_mpi.py <SHOT> <NSTEPS> 1 1

The "1"'s at the end enable checkpointing and a quiet mode. The latter two options can be avoided by setting 0's instead. Diagnostics other than Alcator C-Mod's Hirex-Sr XICS system will be implemented in the future. 
 
 To view the results after running the mpirun command, re-run the same command without mpirun, i.e. use
 
 $ python bsfc_run_mpi.py <SHOT> <NSTEPS> 1 1
 
 ![test](https://user-images.githubusercontent.com/25516628/50047623-0bcbc780-0087-11e9-9c52-3733b9a4f9e6.png)

 
