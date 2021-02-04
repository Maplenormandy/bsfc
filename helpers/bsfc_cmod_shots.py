# -*- coding: utf-8 -*-
"""
Database of C-Mod shots, corresponding atomic lines, times of interest, THT for Hirex-Sr data access, etc..

@author: sciortino
"""
from builtins import str



def get_shot_info(shot, imp_override=None):
    ''' Function to output key information for BSFC fitting of atomic lines  '''

    if shot==1121002022:
        primary_impurity = 'Ar' if imp_override is None else imp_override
        primary_line = 'lya1'
        tbin=5; chbin=40
        t_min=0.7; t_max=0.8
        tht=0
    elif shot==1120914029:    #I-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'lya1'
        tbin=104; chbin=11
        t_min= 1.29; t_max=1.4
        tht=9 ############# tht=0 has 15 chords; tht=9 has 32
    elif shot==1120914036:      # I-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'lya1'
        tbin=104; chbin=11
        #t_min=1.05; t_max=1.27
        t_min= 0.89; t_max=1.05
        tht=5
    elif shot==1101014019:       # EDA H-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'z' #'z' #'z' # 'w'
        #t_min = 0.83; t_max = 1.4 # entire LBO interval
        t_min=1.24; t_max=1.4
        #t_min=1.26; t_max=1.27
        if primary_impurity=='Ar':   # for Ca: THT=0; for Ar: THT=1?
            tht=1
            tbin=1.25 # if set to float, fitSingleBin looks for corresponding bin
            chbin=11 # random 
        else:
           tht=0
           tbin=125; chbin=11
    elif shot==1101014029:      # I-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'w' # 'z' #'w'
        tbin=120; chbin=7  #good
        #tbin=6; chbin=19   # apparently below noise level ?
        #tbin=9; chbin = 4    # very little signal, fit should be thrown out
        #t_min=1.18; t_max=1.3
        t_min=0.78; t_max=1.55 # entire LBO interval
        tht=0
    elif shot==1101014030:    # I-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override  
        primary_line = 'q' #'z' #'x' #'w' #'all' #'z' # 'w'
        #t_min = 1.2; t_max = 1.3
        t_min=1.185; t_max=1.3
        if primary_impurity=='Ar':   # for Ca: THT=0; for Ar: THT=1
            tht=1
            tbin = 6; chbin = 18
        else:
           tht=0
           #tbin=128; chbin=31  # t=1.2695
           tbin=116; chbin=18  # t=1.2095, ~ peak Ca signal
           #tbin=116; chbin=8 # unknown signal comes up near 3.196A only in this channel, motivated shorter lambda bounds
           #tbin=135; chbin=8   # t=1.3115
           ####t_min=0.780; t_max = 1.5  # entire LBO interval
           
    elif shot==1100305019:
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'w'
        # tbin=128; chbin=11
        tbin=116; chbin=18
        t_min=0.98; t_max=1.2
        tht=9
    elif shot==1160506007:
        primary_impurity = 'Ar' if imp_override is None else imp_override
        primary_line = 'w'
        tbin = 46; chbin = 40
        t_min=0.93; t_max=0.99 #counter-current rotation SOC
        #t_min=0.57; t_max=0.63 #co-current rotation LOC
        tht = 0
    elif shot==1150903021:
        primary_impurity = 'Ar' if imp_override is None else imp_override
        primary_line = 'w'
        tbin = 16; chbin = 6
        t_min=0.93; t_max=0.99
        tht = 2
    elif shot==1160920007:
        primary_impurity = 'Ar' if imp_override is None else imp_override
        primary_line = 'lya1'
        tbin = 12; chbin = 4
        t_min=0.81; t_max=0.84
        tht = 0
    elif shot==1101014006:     # L-mode FS
        # for Ca: THT=0; for Ar: THT=1
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'z' #'w' #'all' #'z' #'w'
        if primary_impurity=='Ar':
            tbin=14; chbin=20   # for Ar
            tht=2
        elif primary_impurity=='Ca':
            #tbin=116; chbin=18
            tbin=124; chbin=11   # good for Ca
            tht=0
        #t_min = 0.75; t_max = 1.5  #entire LBO interval
        t_min=1.155; t_max=1.265
    elif shot==1101014011:     # L-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'z' #'w'
        #tbin=116; chbin=18
        tbin=124; chbin=11
        #t_min=0.7; t_max=0.95
        t_min = 0.75; t_max = 1.5  #entire LBO interval
        tht=0
    elif shot==1101014012:     # L-mode FS
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'w' # 'z'
        #tbin=116; chbin=18
        tbin=124; chbin=11
        #t_min=1.150; t_max=1.3
        t_min = 0.75; t_max = 1.5  #entire LBO interval
        tht=0   
    elif str(shot).startswith('1140729'): #1140729021 or shot==1140729023 or shot==1140729030:
        primary_impurity = 'Ca' if imp_override is None else imp_override
        primary_line = 'w'
        tbin=155; chbin=1  #tbin=155 is t=1.43
        #t_min=0.98; t_max=1.2 #1.15
        t_min=1.38; t_max = 1.499
        tht=9
    else:
        # define more lines!
        raise Exception('Times of interest not set for this shot!')

    return primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht
