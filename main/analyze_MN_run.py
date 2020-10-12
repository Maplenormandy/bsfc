import pickle as pkl
import numpy as np
from IPython import embed

def process_MN_run(shot,t_min,t_max,tb,cb,primary_impurity,primary_line,basename,resfile_base):
    '''
    Method to analyze MultiNest output for a single BSFC run.
    Results are stored in a directory/file that can be read by bsfc_ns_shot.

    TODO: analyze multiple lines in the same multinest run and save measurements to different files?
    '''

    with open(f'../bsfc_fits/mf_{shot:d}_tmin{t_min:.2f}_tmax{t_max:.2f}_{primary_line:s}_{primary_impurity:s}.pkl','rb') as f:
        mf=pkl.load(f)

    # load MultiNest output
    good = mf.fits[tb][cb].MN_analysis(basename)
    
    # process samples and weights
    samples = mf.fits[tb][cb].samples
    sample_weights = mf.fits[tb][cb].sample_weights

    
    # if line_name is None:
    #     # if user didn't request a specific line, assume that primary line is of interest
    #     line_name = mf.primary_line

    # try:
    #     line_id = np.where(mf.fits[tb][cb].lineModel.linesnames==line_name)[0][0]
    #     print(f'Analyzing {line_name} line')
    # except:
    #     raise ValueError('Requested line cannot be found in MultiNest output!')
    
    for line_name in mf.fits[tb][cb].lineModel.linesnames:
        if line_name not in mf.nofit and line_name not in ['m','s','t']: # don't save lines that were not fitted or are not interesting
            # collect results from fits
            line_id = np.where(mf.fits[tb][cb].lineModel.linesnames==line_name)[0][0]
            modelMeas = lambda x: mf.fits[tb][cb].lineModel.modelMeasurements(x, line=line_id)
            measurements = np.apply_along_axis(modelMeas, 1, samples)
            moments_vals = np.average(measurements, 0, weights=sample_weights)
            moments_stds = np.sqrt(np.average((measurements-moments_vals)**2, 0, weights=sample_weights))
            res_fit = [np.asarray(moments_vals), np.asarray(moments_stds)]

            # save measurement results to checkpoint file
            resfile = resfile_base + f'_{line_name:s}line.pkl'
            with open(resfile,'wb') as f:
                pkl.dump(res_fit,f)

