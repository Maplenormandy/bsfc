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
    elif shot==1120914036:
        primary_impurity = 'Ca'
        primary_line = 'lya1'
        tbin=104; chbin=11
        t_min=1.05; t_max=1.27
        tht=5
    elif shot==1101014019:
        primary_impurity = 'Ca'
        primary_line = 'w'
        #tbin=128; chbin=11
        #tbin=111; chbin=31  #problematic: works with 1000 steps, but not 25000
        tbin=111; chbin=15   #problematic
        t_min=1.24; t_max=1.4
        tht=0
    elif shot==1101014029:
        primary_impurity = 'Ca'
        primary_line = 'w'
        #tbin=128; chbin=11  #good
        #tbin=6; chbin=19   # apparently below noise level ?
        tbin=9; chbin = 4
        t_min=1.17; t_max=1.3
        tht=0
    elif shot==1101014030:
        primary_impurity = 'Ca'
        primary_line = 'w'
        # tbin=128; chbin=11
        tbin=116; chbin=18
        t_min=1.17; t_max=1.3
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
        t_min=0.93; t_max=0.99
        tht = 0
    else:
        # define more lines!
        raise Exception('Times of interest not set for this shot!')
        

    return primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht
