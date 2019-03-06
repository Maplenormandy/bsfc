import sys
import cPickle as pkl
import os
import numpy as np
import glob

# inputs to identify MomentFitter file and bin to fit
shot=int(sys.argv[1])
t_min = float(sys.argv[2])
t_max = float(sys.argv[3])
tb = int(sys.argv[4])
cb = int(sys.argv[5])
verbose = bool(sys.argv[6])
rank=int(sys.argv[7])

#print "Fitting bin [%d,%d] with MultiNest...."%(tb,cb)

# load moment fitter setup 
with open('../bsfc_fits/mf_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'rb') as f:
    mf=pkl.load(f)

# set up directory for MultiNest to run
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains%s/c-.'%rank )
chains_dir = os.path.dirname(basename)

if not os.path.exists(chains_dir):
    # if chains directory for this worker/rank does not exist, create it
    os.mkdir(chains_dir)
else:
    # if chains directory exists, make sure it's empty
    fileList = glob.glob(basename+'*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            pass
    
# do fit in the directory of 'basename'
mf.fitSingleBin(tbin=tb, chbin=cb,NS=True, n_hermite=3, n_live_points=50,
                                sampling_efficiency=0.3, verbose=verbose, basename=basename)

# load MultiNest output
mf.fits[tb][cb].NS_analysis(basename)

# process samples and weights
samples = mf.fits[tb][cb].samples
sample_weights = mf.fits[tb][cb].sample_weights
measurements = np.apply_along_axis(mf.fits[tb][cb].lineModel.modelMeasurements, 1, samples)
moments_vals = np.average(measurements, 0, weights=sample_weights)
moments_stds = np.sqrt(np.average((measurements-moments_vals)**2, 0, weights=sample_weights))
res_fit = [np.asarray(moments_vals), np.asarray(moments_vals)]

# checkpoints and case_dir directories must be created externally to this script!
checkpoints_dir ='checkpoints/'
case_dir = '{:d}_tmin{:.2f}_tmax{:.2f}/'.format(shot,t_min,t_max)
file_name = 'moments_{:d}_bin{:d}_{:d}.pkl'.format(shot,tb,cb)
resfile = checkpoints_dir + case_dir + file_name

# save measurement results to checkpoint file
with open(resfile,'wb') as f:
    pkl.dump(res_fit,f)
