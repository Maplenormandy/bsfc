#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.signal import order_filter
from scipy.stats.mstats import mquantiles
import pdb
plt.ion()

from scipy.optimize import curve_fit
import MDSplus as mds

from scipy.io import readsav
shots=[175861] 
'''
shots=[175849,175850,175851,175852,175853,175854,175855,175856,175857,175858,175859,175860,175861,175862,
       175863,175864,175865,175866,175867,175868,175882,175883,175884,175885,175886,175887,175888,175889,
       175898,175899,175900,175901,175902,175903,175904]
'''

los_list= ['T11']
#los_list =  ['T%.2d'%d for d in range(1,49)]+['V%.2d'%d for d in range(1,33)]

mdsserver = 'atlas.gat.com'
import MDSplus
MDSconn = MDSplus.Connection(mdsserver)
#for nch in range(1,48): 
    #try:
        #print median(MDSconn.get(r'\IONS::TOP.CER.CERAUTO.%s.CHANNEL%.2d:R'%(system,nch)))
    #except:
        #pass
     
lines = []
 
for shot in shots:
    MDSconn.openTree('D3D', shot)
 
    for los in los_list:
        print(shot, los)
        cer_data = readsav('./data/shot%d%s.sav'%(shot,los))
 
        data = cer_data['chord_data'][0]
         
        #if los != 'T17':
            #continue
        #R =  MDSconn.get('.CER.%s:DATE_LOADED'%analysis_type).data()
 
        nch = int(los[1:])
        system = 'TANGENTIAL' if los[0] == 'T' else 'VERTICAL'
        #print r'\IONS::TOP.CER.CERAUTO.%s.CHANNEL%s'%(system,nch)
        #try:
        nbi30R = MDSconn.get(r'_x=\D3D::TOP.NB.NB30R:PINJ_30R').data()
        nbi30R_t = MDSconn.get(r'dim_of(_x)').data()
 
        nbi30L = MDSconn.get(r'_x=\D3D::TOP.NB.NB30L:PINJ_30L').data()
        nbi30L_t = MDSconn.get(r'dim_of(_x)').data()
        
        ##import IPython
        ##IPython.embed()
        
        
        shot,chord,dark_shot,type,channels,idx,white,t_start,t_integ,tg,data_grp,bg_grp,wl,comments,shot_time,timing_def,timing_mode,bg_mode,camera_type,gain,bandwidth,binning,raw,data = data
 
        if wl < 4050 or wl > 4100:
            continue

        t_start = t_start[tg<=data_grp.max()]
        t_integ = t_integ[tg<=data_grp.max()]
        time_vec = t_start + old_div(t_integ,2.0)

        corrupted = data - order_filter(data,np.ones((1,5)),1)
        corrupted[:,0] = 0
        corrupted[:,-1] = 0
        corrupted  = corrupted > mquantiles(corrupted,0.999)
    
        '''
        # get background
        for it in np.arange(data.shape[0]):
            data[it,corrupted[it]] = np.interp(np.where(corrupted[it])[0],np.where(~corrupted[it])[0],data[it,~corrupted[it]])
            bcg = mquantiles(data[it],.3)
            data[it] -= bcg
        '''
        
        #import IPython
        #IPython.embed()

        # plot data matrix
        plt.figure(figsize=(6,12))
        plt.imshow(data, interpolation='nearest', aspect='auto',vmin=0,vmax=50, origin='lower', extent=(0,1,old_div(t_start[0],1000),old_div((t_start[-1]+t_integ[-1]),1000)));plt.colorbar()
        plt.ylim(1,6)
        plt.ylabel('time [s]')
        plt.xlabel('Wavelength range? ')
        #plt.xlim(.3,.7)
        plt.axvline(.5,c='w')
        plt.axhline(2,c='w')
        plt.axhline(3,c='w')
        plt.axhline(4.5,c='w')
        plt.title(str(shot)+'  '+los+'  '+str(wl) )
        
        '''
        #import IPython
        #IPython.embed()
        u,s,v = np.linalg.svd(data,full_matrices=0,)

        #plt.plot(nbiL30_t/1e3,nbiL30/2e7)
         
        nbi_on = np.interp(t_start+t_integ/2,  nbi30L_t,nbi30L) > 1e5
        line = np.sign(np.nanmean(u[:,2] )-np.nanmedian(u[:,2] ))*u[:,2]*s[2]
        line-= np.nanmedian(line)
         
        line_t = (t_start+t_integ/2)/1e3
        line[~nbi_on]=0

        line,line_t = line[nbi_on],line_t[nbi_on]
        plt.figure()
        plt.plot(line_t,line,'o-')
        plt.xlim(1,6)
        plt.title(str(shot)+'  '+los+'  '+str(wl) )
        #plt.savefig('line'+str(shot)+'_'+los)
        #plt.clf()
        
        lines.append((line))
        '''

        '''
        plt.figure()
        u[:,2][~nbi_on] = np.nan
        plt.plot((t_start+t_integ/2)/1e3,np.sign(np.nanmean(u[:,2] ))*u[:,2]*s[2] ,'o-')
         
        plt.step(t_start/1e3,np.sign(np.mean(u[:,2]-np.median(u[:,2])))*(u[:,2]-np.median(u[:,2])),where='post')
        plt.axvline(2,c='k')
        plt.axvline(3,c='k')
        plt.axvline(4.5,c='k')
        plt.xlim(1.8,6)
        #plt.ylim(-np.std(u[:,2]),None)
         
        plt.plot(nbi30L_t,nbi30L)
 
        #plt.show()
         '''
        
        # definition of Gaussian fitting function:
        def gauss(x,  A, mu, sigma):
            return A*np.exp(old_div(-(x-mu)**2,(2.*sigma**2)))/np.sqrt(2*np.pi)/sigma
        
        # Define time range to fit
        t_min = 2000.0   # ms
        t_max = 3000.0 #ms
        
        tmin_idx = np.argmin(np.abs(time_vec - t_min))
        tmax_idx = np.argmin(np.abs(time_vec - t_max))
        time_sel = time_vec[tmin_idx:tmax_idx]

        # initialization
        x = np.linspace(0,1,data.shape[1])
        coeffs = np.zeros((len(time_sel), 3)) 
        coeffs_err = np.zeros_like(coeffs) 
        
        ii = 0
        for it in np.arange(tmin_idx,tmax_idx):

            bcg = mquantiles(data[it],.3)
            coeffs[ii], var_matrix = curve_fit(gauss,x,data[it]-bcg,p0=(20,.3,0.01), 
                                               sigma = np.sqrt(np.maximum(5,data[it])) ,
                                               bounds=(  (0,.15,0.01), (np.infty, .4,0.05)))
                                               #bounds=(  (0,.45,0.01), (np.infty, .55,0.05)))

            if var_matrix is not None:
                coeffs_err[ii] = np.sqrt(np.diag(var_matrix)) 
        
            ii +=1

        # plot specific fit
        plt.figure(3)
        plt.plot(x, gauss(x,*coeffs[ii-1])+bcg)
        plt.plot(x, data[it], '.')
        plt.axhline(bcg, color='k')
        
        # NBI on vs. off
        nbi_on = np.interp(time_sel,  nbi30L_t, nbi30L) > 1e5
        plt.figure(); plt.plot(time_sel, nbi_on)
        plt.xlabel('time [ms]')
        plt.ylabel('NBI on vs. off')
        
        # normalization of NBI trace
        nbi_tmin_idx = np.argmin(np.abs(nbi30L_t - t_min))
        nbi_tmax_idx = np.argmin(np.abs(nbi30L_t - t_max))

        nbi30L_t_cut = nbi30L_t[nbi_tmin_idx:nbi_tmax_idx]
        nbi30L_cut = nbi30L[nbi_tmin_idx:nbi_tmax_idx]
        
        ####
        # Gaussian parameters: A, mu, sigma
        plt.figure(); plt.step(time_sel, old_div((coeffs[:,0]- min(coeffs[:,0])),max(coeffs[:,0]))); 
        #plt.ylim([coeffs[:,0].min(),None])
        plt.plot(nbi30L_t_cut,nbi30L_cut/max(nbi30L_cut) *plt.gca().get_ylim()[1], alpha=0.5, label='NBI')
        plt.xlabel('time [ms]')
        plt.ylabel('Gaussian amplitude [A.U.]')
        plt.legend().draggable()

        plt.figure(); plt.step(time_sel,  old_div((coeffs[:,1]- min(coeffs[:,1])),max(coeffs[:,1])));
        #plt.ylim([coeffs[:,1].min(),None])
        plt.plot(nbi30L_t_cut,nbi30L_cut/max(nbi30L_cut) *plt.gca().get_ylim()[1], alpha=0.5, label='NBI')
        plt.xlabel('time [ms]')
        plt.ylabel('Gaussian mean [A.U.]')
        plt.legend().draggable()

        plt.figure(); plt.step(time_sel,  old_div((coeffs[:,2]- min(coeffs[:,2])),max(coeffs[:,0])));
        #plt.ylim([coeffs[:,2].min(),None])
        plt.plot(nbi30L_t_cut,nbi30L_cut/max(nbi30L_cut) *plt.gca().get_ylim()[1], alpha=0.5, label='NBI')
        plt.xlabel('time [ms]')
        plt.ylabel('Gaussian width [A.U.]')
        plt.legend().draggable()

        #plt.step(t_start,coeffs[:,0])
        
 
 

'''
plt.imshow(corrupted, interpolation='nearest', aspect='auto',vmin=0,vmax=1, origin='lower', extent=(0,1,t_start[0]/1000,(t_start[-1]+t_integ[-1])/1000));plt.colorbar() #,show()
 
plt.imshow(data, interpolation='nearest', aspect='auto',vmin=0,vmax=200, origin='lower', extent=(0,1,t_start[0]/1000,(t_start[-1]+t_integ[-1])/1000));plt.colorbar()
plt.axhline(2,c='w')
plt.axhline(3,c='w')
plt.axhline(4.5,c='w')
plt.show()
 
plt.imshow(data, interpolation='nearest', aspect='auto',vmin=0,vmax=200, origin='lower');plt.colorbar()
plt.axhline(2,c='w')
plt.axhline(3,c='w')
plt.axhline(4.5,c='w')
#plt.show()
 
'''
 
 
 
 
 
#dtype((numpy.record, [(('shot', 'SHOT'), '>i4'),
                      #(('chord', 'CHORD'), 'O'),
                      #(('dark_shot', 'DARK_SHOT'), '>i4'),
                      #(('type', 'TYPE'), '>i2'),
                      #(('channels', 'CHANNELS'), '>i2'),
                      #(('idx', 'IDX'), '>i4'),
                      #(('white', 'WHITE'), '>i2'), 
                      #(('t_start', 'T_START'), 'O'), 
                      #(('t_integ', 'T_INTEG'), 'O'),
                      #(('tg', 'TG'), 'O'),
                      #(('data_grp', 'DATA_GRP'), 'O'),
                      #(('bg_grp', 'BG_GRP'), 'O'),
                      #(('wl', 'WL'), '>f4'),
                      #(('comments', 'COMMENTS'), 'O'),
                      #(('shot_time', 'SHOT_TIME'), 'O'),
                      #(('timing_def', 'TIMING_DEF'), 'O'),
                      #(('timing_mode', 'TIMING_MODE'), '>i2'),
                      #(('bg_mode', 'BG_MODE'), '>i2'),
                      #(('camera_type', 'CAMERA_TYPE'), '>i2'),
                      #(('gain', 'GAIN'), '>i2'),
                      #(('bandwidth', 'BANDWIDTH'), '>i2'),
                      #(('binning', 'BINNING'), '>i2'),
                      #(('raw', 'RAW'), 'O'),
                      #(('data', 'DATA'), 'O')]))
