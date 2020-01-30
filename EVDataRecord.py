# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:12:25 2019

@author: Orlando Timmerman
"""
import numpy as np
import os

def readdata():

    folder = input('Folder directory: ')    #directory T1,T2,etc.
    os.chdir(folder)    #navigates to correct directory
    all_files = os.listdir(folder)  #list of files in directory
    numfiles = len(all_files)
    for i in range(numfiles):   #iterates through each file
        filename = all_files[i]

        f = open(filename, 'r')
        lines = f.readlines()   #list of lines in textfile
        numlines = len(lines)
        data = []
          
        sample_rate = float((lines[3])[16:-1])  #extracts TRACER sample rate
        #sample_count = (lines[4])[15:-1]
        
        for i in range(numlines):
            if i > 8 and i < (numlines-1):  #creates list of voltage output, calls it data (diregards preamble)
                data.append(((lines[i])[-7:-1])[0])
                
        index_list = []
        TP_list = []
        TP = 0
        period = 0
        SEM = 0
    
        first = True
        #find first value of group with non-zero voltage (lightgate activated) and report index
        for j in range(len(data)):
            if float(data[j]) != 0 and first == True:
                index_list.append(j)
                first = False
                if len(index_list) > 1: #to avoid null initial
                    TP_list.append(abs((index_list[-1:])[0]-(index_list[-2:-1])[0]))
            elif float(data[j]) == 0:
                first = True
    
        for k in range(len(TP_list)):   #sums all time period indices
            TP += TP_list[k]

        period = TP/(len(TP_list)*sample_rate)  #calculates average period
    
        for l in range(len(TP_list)):   #calculates SEM
            SEM += (TP_list[l]-period)**2
        SEM = 1/(len(TP_list)-1)
    
        #prints result N.B. NOT IN DIRECTORY ORDER!
        print('Filename: ',filename)
        print('Average period: {0:.6}s'.format(period))
        print('Standard Error on Mean: ',SEM)
    

readdata()