# -*- coding: utf-8 -*-
"""
Database of C-Mod shots, corresponding atomic lines, times of interest, THT for Hirex-Sr data access, etc..

@author: sciortino
"""



def get_shot_info(shot):
    ''' Function to output key information for BSFC fitting of atomic lines  '''

    if shot==1121002022:
        primary_impurity = 'Ar'
        primary_line = 'lya1'
        tbin=5; chbin=40
        t_min=0.7; t_max=0.8
        tht=0
    elif shot==1120914029:    #I-mode FS
        primary_impurity = 'Ca'
        primary_line = 'lya1'
        tbin=104; chbin=11
        t_min= 1.29; t_max=1.4
        tht=9 ############# tht=0 has 15 chords; tht=9 has 32
    elif shot==1120914036:      # I-mode FS
        primary_impurity = 'Ca'
        primary_line = 'lya1'
        tbin=104; chbin=11
        #t_min=1.05; t_max=1.27
        t_min= 0.89; t_max=1.05
        tht=5
    elif shot==1101014019:       # EDA H-mode FS
        primary_impurity = 'Ca'
        primary_line = 'w'
        tbin=128; chbin=11
        #tbin=111; chbin=31  #problematic: works with 1000 steps, but not 25000
        #tbin=111; chbin=15   #problematic
        t_min=1.24; t_max=1.4
        #t_min=1.26; t_max=1.27
        tht=0
    elif shot==1101014029:      # I-mode FS
        primary_impurity = 'Ca'
        primary_line = 'w'
        tbin=120; chbin=7  #good
        #tbin=6; chbin=19   # apparently below noise level ?
        #tbin=9; chbin = 4    # very little signal, fit should be thrown out
        t_min=1.18; t_max=1.3
        tht=0
    elif shot==1101014030:    # I-mode FS
        primary_impurity = 'Ca'
        primary_line = 'w'
        #tbin=128; chbin=31  # t=1.2695
        tbin=116; chbin=18  # t=1.2095, ~ peak signal
        t_min=1.185; t_max=1.3
        tht=0
    elif shot==1100305019:
        primary_impurity = 'Ca'
        primary_line = 'w'
        # tbin=128; chbin=11
        tbin=116; chbin=18
        t_min=0.98; t_max=1.2
        tht=9
    elif shot==1160506007:
        primary_impurity = 'Ar'
        primary_line = 'w'
        tbin = 46; chbin = 27
        t_min=0.93; t_max=0.99 #counter-current rotation SOC
        #t_min=0.57; t_max=0.63 #co-current rotation LOC
        tht = 0
    elif shot==1150903021:
        primary_impurity = 'Ar'
        primary_line = 'w'
        tbin = 16; chbin = 6
        t_min=0.93; t_max=0.99
        tht = 2
    elif shot==1160920007:
        primary_impurity = 'Ar'
        primary_line = 'lya1'
        tbin = 16; chbin = 4
        t_min=0.81; t_max=0.84
        tht = 0
    elif shot==1101014006:     # L-mode FS
        primary_impurity = 'Ca'
        primary_line = 'w'
        tbin=116; chbin=18
        t_min=1.155; t_max=1.265
        tht=0
    elif str(shot).startswith('1140729'): #1140729021 or shot==1140729023 or shot==1140729030:
        primary_impurity = 'Ca'
        primary_line = 'w'
        tbin=155; chbin=15  #tbin=155 is t=1.43
        #t_min=0.98; t_max=1.2 #1.15
        t_min=1.38; t_max = 1.499
        tht=9
    else:
        # define more lines!
        raise Exception('Times of interest not set for this shot!')


    return primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht
