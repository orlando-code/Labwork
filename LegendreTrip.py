#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:25:20 2020

@author: orlandotimmerman
"""

#####   GENERATE TABLE OF LEGENDRE TRIPLE-SQUARES   #####

def posint(userinput):
    '''Ensures user input is a positive integer.'''
    #while function is called
    while True:
        #if this block is true
        try:
            tocheck = int(input(userinput))
            #checking that snumber is positive
            if tocheck <= 0:    
                print('This number is not positive.')
                continue
        #if not, return this, and call input request again
        except ValueError:
            print('Your response was not an integer.')
        #when correct, return value and exit function
        else:
            return tocheck
        

def gen_nums(max_num):
    '''Generates array of numbers which cannot be produced by the sum of
    three squared integers.'''
    
    vals = []
    for a in range(100):
        for b in range(100):
            val = 4**a * (8*b + 7)
            if val > max_num:
                break
            else:
                vals.append((a,b,val))
    
    vals.sort(key=lambda tup: tup[2])
    return vals
    

def gen_table(vals):
    '''Formats values into regular table.'''
    
    table_head = '''\n\n   
 ---------------------  
   integers             
    a    b       value  
 ---------------------'''
    print(table_head)
    
    
    for i in range(len(vals)):
        a = vals[i][0]
        b = vals[i][1]
        val = vals[i][2]
        
        print('    {}{}{}{}|   {}'.format(a,(5-len(str(a)))*' ',b,(4-len(str(b)))*' ',val))


max_num = posint('Enter the desired number limit: ')
values = gen_nums(max_num)
gen_table(values)

