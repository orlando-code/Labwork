#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:47:17 2020

@author: orlandotimmerman
"""

# =============================================================================
# PROGRAM FOR MODELLING ORBITS
# =============================================================================


#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool
from matplotlib.ticker import EngFormatter
import os
import sys


# =============================================================================
# GLOBAL CONSTANTS - given unmistakeable capitalisation
# =============================================================================

GRAV = 6.67e-11     #gravitational constant
M_E = 5.97e24       #mass of earth (kg)
M_R = 500           #mass of rocket (kg)
M_M = 7.35e22       #mass of moon (kg)
R_E = 6371e3        #radius of earth (m)
R_M = 1736e3        #radius of moon (m)
D_M = 384.4e6       #earth-moon distance (m)

# =============================================================================
# FUNCTIONS ENSURING CORRECT INPUT
# =============================================================================

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
        
        
def posinteven(userinput):
    '''Ensures user input is a positive even integer.'''
    #while function is called
    while True:
        #if this block is true
        try:
            tocheck = int(input(userinput))
            #checking that number is positive
            if tocheck <= 0:    
                print('This number is not positive.')
                continue
            elif tocheck % 2 ==1:
                print('This number is not even.')
                continue
        #if not, return this, and call input request again
        except ValueError:
            print('Your response was not an integer.')
        #when correct, return value and exit function
        else:
            return tocheck
        
        
def posflt(userinput):
    '''Ensures user input is a positive floating point number.'''
    while True: #while function called, if this block is true
        try:
            tocheck = float(input(userinput))
            if tocheck < 0:
                print('This number is not positive.')
                continue
        except ValueError:  #if not, return this, and call input request again
            print('That is not a floating point number.')
        else:   #when correct, return value and exit function
            return tocheck 


def flt(userinput):
    '''Ensures user input is a floating point number.'''
    while True: #while function called, if this block is true
        try:
            tocheck = float(input(userinput))
        except ValueError:  #if not, return this, and call input request again
            print('That is not a floating point number.')
        else:   #when correct, return value and exit function
            return tocheck 
    
    
def yesno(userinput):
    '''Demands a yes/no response from user and stops at nothing to get it.'''
    while True: #while function called
        try:
            return strtobool(input(userinput).lower())
        except ValueError:
            print('Please answer with ''yes'' or ''no''.')
           
            
def ticker(i,final_i):
    '''Produces loading bar with percentage counter. I spent far too long on this...'''
    wheel = ['-','/','|','\\','-']          #different ticker orientations
    perc = (((i)/final_i)*100)              #calculates percent to nearest integer
    stars = ' *' * int(((perc+1)//10))      #determines no. of stars
    dashes = ' -' * (10-(len(stars)//2))    #determines no. of dashes
    index_1 = int(i % 5)                    #allows wheels to rotate with opposite orientations
    index_2 = 4-index_1
    #Fancy loading formatting! Replaces readout continuously
    sys.stdout.write('\r '+stars+dashes+'    '+wheel[index_2]+'  '\
                     +str(round(perc))+'%'+'  '+wheel[index_1]+'    '+dashes+stars)        
            
    
def greet():
    '''Displays greeting message.'''
    print('\nGood day!\n')
    print('''I hope you\'re well! I\'m sure you\'ll be even better after seeing
some supercool orbit investigations!''')
    
    
def myfolder():
    '''Gets directory. If folder exists, navigates to it. If it doesn't exist, folder is created.'''
    print('''\nGraphs produced by program will be saved in folder called \'xe_18037_ex4_graphs\'.
          \nThey will also be displayed as figures when the user quits the program.''')   #info and folder name
    
    Response = 0
    while Response == 0:            #keep giving input option until user is happy
        path = input('Please enter your chosen directory in which to place folder: ')       #enter directory path
        repeat = 'You have chosen: {}'.format(path) + '\nIs this path correct? (y/n): '     #check if correct
        Response = yesno(repeat)    #updates response
        
    folder = os.path.join(path,'xe_18037_ex4_graphs')   #concatenates full path name
    if os.path.exists(folder) != True:                  #check if folder already exists (files will be overwritten)
        os.makedirs(folder)                             #if folder doesn't exist, new one made
        print('Folder created.\nGraphs which are re-created will be overwritten to latest instance.')
        os.chdir(folder)
    else:                           #if folder already exists
        os.chdir(folder)            #changes directory to folder
        print('''Navigated to correct folder.
              \nGraphs which are re-created from previous runs will be overwritten in folder to latest instance.''')
        
        
# =============================================================================
# INPUT FUNCTIONS
# =============================================================================
    
def trial_error_input(variable):
    '''Takes user input for trial and error function.'''
    
    num_tries = posint('Enter number of trials: ')
    while num_tries > 10:   #caps number of possible trials based on processing and visualisation
        print('That\'s too many tries to clearly visualise on a graph!')
        num_tries = posint('\nEnter number of trials (10 or fewer): ')
    
    possible_vars = ['initial x position','initial y position','initial x velocity','initial y velocity','step size']
    val_range = flt('''Enter range of values (in S.I. units) for varying {}.
N.B. Negative ranges decrease initial values.
Range: '''.format(possible_vars[int(variable)-1]))      #reminds use of their variable choice
    
    return num_tries, val_range
    

def initial_values_input():
    '''Takes user input for initial values of orbit parameters.'''
    
    x_0 = flt('Enter rocket\'s initial x position (km): ')
    y_0 = flt('Enter rocket\'s initial y position (km): ')
    
    in_earth = np.sqrt(x_0**2+y_0**2)       #earth radius values
    in_moon = np.sqrt(x_0**2+(y_0-D_M)**2)  #moon radius values
    
    if in_earth < R_E/1e3:  #ensures not starting within earth
        x_0 = flt('Your rocket\'s initial position was within the Earth! \nEnter rocket\'s initial x position (km): ')
        y_0 = flt('Enter rocket\'s initial y position (km): ')
    if in_moon < R_M/1e3:   #ensures not starting within moon
        x_0 = flt('Your rocket\'s initial position was within the Moon! \nEnter rocket\'s initial x position (km): ')
        y_0 = flt('Enter rocket\'s initial y position (km): ')
        
    v_x_0 = flt('Enter rocket\'s initial x velocity (km/s): ')
    v_y_0 = flt('Enter rocket\'s initial y velocity (km/s): ')
    
    return x_0*1e3, y_0*1e3, v_x_0*1e3, v_y_0*1e3   #returns values in SI units


def simp_inputting():
    '''Takes user input for overarching simulation parameters.'''
    
    runtime = posint('Enter simulation run time: ')    
    deltat = posflt('Enter timestep: ')
    
    return runtime, deltat


def change_var():
    '''Menu function to investigate effect of changing parameters on resulting orbit.'''
    
    VarInput = '0'
    model = yesno('[y/n] Consider moon\'s gravity? ')
    text_art()      #prints out earth diagram
    while VarInput not in ['1','2','3','4','5']:
        varquestion = '''
    Which variable would you like to change?
    Enter a choice - "1" (rocket\'s initial x position),
                     "2" (rocket\'s initial y position),
                     "3" (rocket\'s initial x velocity),
                     "4" (rocket\'s initial y velocity),
                     "5" (step size): '''

        VarInput = input(varquestion).lower()
        if VarInput == '1' or VarInput == '2' or VarInput == '3' or VarInput == '4' or VarInput == '5': #ensure entters valid variable
            return VarInput, model
        elif VarInput not in ['1','2','3','4','5']:   #must enter '1', '2', '3', '4', or '5'
            print('Choice not valid.')

            
# =============================================================================
# RUNGE-KUTTA FUNCTIONS
# =============================================================================
    
def F1(v_x):
    return v_x

def F2(v_y):
    return v_y

def F3(x,y,model):
    den_e = (x**2+y**2)**(3/2)                      #earth acceleration equation denominator
    if model == False:    #if not including moon
        num_e = -GRAV*M_E*x                         #earth acceleration equation numerator
        acc = num_e/den_e                           #resulting acceleration
    elif model == True:  #if including moon
        den_m = (x**2+(y-D_M)**2)**(3/2)            #moon acceleration equation denominator
        acc = -GRAV*(((M_E*x)/den_e)+(M_M*x/den_m)) #resulting acceleration
    return acc

def F4(x,y,model):
    den_e = (x**2+y**2)**(3/2)                      #earth acceleration equation denominator  
    if model == False:    #if not including moon
        num_e = -GRAV*M_E*y
        acc = num_e/den_e
    elif model == True:  #if including moon
        den_m = (x**2+(y-D_M)**2)**(3/2)            #moon acceleration equation denominator 
        acc = -GRAV*(((M_E*y)/den_e)+(M_M*(y-D_M)/den_m))   #resulting acceleration
    return acc
    

def calc_k(x,y,v_x,v_y,deltat,model):
    '''Function to calculate 4th order Runge-Kutta coefficients as vectors.'''
    
    #assign empty four-element vectors for RK4 coefficient assignment
    k_x = np.zeros((4))
    k_y = np.zeros((4))
    k_vx = np.zeros((4))
    k_vy = np.zeros((4))
    
    #k1 values 
    k_x[0] = F1(v_x)
    k_y[0] = F2(v_y)
    k_vx[0] = F3(x,y,model)
    k_vy[0] = F4(x,y,model)
    
    #k2 values 
    k_x[1] = F1(v_x + (0.5 * deltat * k_vx[0]))
    k_y[1] = F2(v_y + (0.5 * deltat * k_vy[0]))
    k_vx[1] = F3(x + (0.5 * deltat * k_x[0]), y + (0.5 * deltat * k_y[0]),model)
    k_vy[1] = F4(x + (0.5 * deltat * k_x[0]), y + (0.5 * deltat * k_y[0]),model)
    
    #k3 values 
    k_x[2] = F1(v_x + (0.5 * deltat * k_vx[1]))
    k_y[2] = F2(v_y + (0.5 * deltat * k_vy[1]))
    k_vx[2] = F3(x + (0.5 * deltat * k_x[1]), y + (0.5 * deltat * k_y[1]),model)
    k_vy[2] = F4(x + (0.5 * deltat * k_x[1]), y + (0.5 * deltat * k_y[1]),model)
    
    #k4 values 
    k_x[3] = F1(v_x + (deltat * k_vx[2]))
    k_y[3] = F2(v_y + (deltat * k_vy[2]))
    k_vx[3] = F3(x + (deltat * k_x[2]), y + (deltat * k_y[2]),model)
    k_vy[3] = F4(x + (deltat * k_x[2]), y + (deltat * k_y[2]),model)
    
    return k_x,k_y,k_vx,k_vy

    
def RK4_orbit(model,filename, runtime=100000, deltat=100, x_0=-(R_E+408e3), y_0=0, v_x_0=0, v_y_0=7.66e3):
    '''Calculates orbit using RK4 method.'''
    
    iterations = int(runtime/deltat)    #number of iterations
    #initialise empty arrays
    x_vals = []
    y_vals = []
    v_x_vals = []
    v_y_vals = []
    t_vals = [0] 
    
    #assign initial values to those inputted; otherwise use default
    x_vals.append(x_0)
    y_vals.append(y_0)
    v_x_vals.append(v_x_0)
    v_y_vals.append(v_y_0)
    r_earth_vals = [np.sqrt(x_vals[0]**2+y_vals[0]**2)]         #initial earth radius
    r_moon_vals = [np.sqrt(x_vals[0]**2+(y_vals[0]-D_M)**2)]    #initial moon radius
    
    #record number of earth or moon collisions for 'mission_report' function
    earth_ticker = 0
    moon_ticker = 0
    
    for i in range(iterations):
        ticker(i,iterations)
    
        x = x_vals[i]       #assign variable for each iteration
        y = y_vals[i]
        v_x = v_x_vals[i]
        v_y = v_y_vals[i]
    
        r_earth = np.sqrt(x**2+y**2)        #distance from centre of earth
        r_moon = np.sqrt(x**2+(y-D_M)**2)   #distance from centre of moon
        r_earth_vals.append(r_earth)
        r_moon_vals.append(r_moon)
    
        k_x,k_y,k_vx,k_vy = calc_k(x,y,v_x,v_y,deltat,model)    #calculate R-K4 coefficients
        
        #finds stopping point of calculation
        if r_earth > R_E and r_moon > R_M:  #if no collision, append successive values
            x_vals.append(x_vals[i] + (deltat/6)*(k_x[0]+2*k_x[1]+2*k_x[2]+k_x[3]))
            y_vals.append(y_vals[i] + (deltat/6)*(k_y[0]+2*k_y[1]+2*k_y[2]+k_y[3]))
            v_x_vals.append(v_x_vals[i] + (deltat/6)*(k_vx[0]+2*k_vx[1]+2*k_vx[2]+k_vx[3]))
            v_y_vals.append(v_y_vals[i] + (deltat/6)*(k_vy[0]+2*k_vy[1]+2*k_vy[2]+k_vy[3]))
            t_vals.append(t_vals[i] + deltat)
        else:   #if collides, stop iteration and return information
            crash_vel = np.sqrt(v_x**2+v_y**2)/1e3      #km/s
            crash_time = t_vals[-1]/3600                #hours
            if r_earth < R_E:   #collided with earth
                print('\n\nThe rocket crashed into the Earth, terminating simulation prematurely.')
                print('impact velocity = {:.3f} km/s| impact time = {:.3f} hours'.format(crash_vel,crash_time))
                earth_ticker += 1
            elif r_moon < R_M:  #collided with moon
                print('\n\nThe rocket crashed into the Moon, terminating simulation prematurely.')
                print('impact velocity = {:.3f} km/s| impact time = {:.3f} hours'.format(crash_vel,crash_time))
                moon_ticker += 1
            break   #stop execution of iteration
            
    return x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename

    
# =============================================================================
# EXTENSION FUNCTIONS
# =============================================================================
    
    
def euler_calc(model, runtime, deltat,x_0=-(R_E+408e3), y_0=0, v_x_0=0, v_y_0=7.66e3, filename='Euler orbit.pdf'):
    '''Calculates orbit using Euler (1st order Runge-Kutta) method for comparison with RK4.'''
    
    iterations = int(runtime/deltat)    #number of iterations
    
    #initialise empty arrays
    x_vals = []
    y_vals = []
    v_x_vals = []
    v_y_vals = []
    t_vals = [0]
    r_earth_vals = []
    r_moon_vals = []

    #assign initial values to those inputted; otherwise use default
    x_vals.append(x_0)
    y_vals.append(y_0)
    v_x_vals.append(v_x_0)
    v_y_vals.append(v_y_0)
    
    #record number of earth or moon collisions for 'mission report' function
    earth_ticker = 0
    moon_ticker = 0
    
    for i in range(iterations):
        ticker(i,iterations)
        
        #assign variable for each iteration
        x = x_vals[i]       
        y = y_vals[i]
        v_x = v_x_vals[i]
        v_y = v_y_vals[i]
        
        #calculate first-order RK coefficients
        e_k_x = F1(v_x)
        e_k_y = F2(v_y)
        e_k_vx = F3(x,y,model)
        e_k_vy = F4(x,y,model)
    
        r_earth = np.sqrt(x**2+y**2)
        r_moon = np.sqrt(x**2+(y-D_M)**2)
        r_earth_vals.append(r_earth)        #checks whether within earth
        r_moon_vals.append(r_moon)          #checks whether within moon
        
        #finds stopping point of function
        if r_earth > R_E and r_moon > R_M:  #if no collision with earth or moon, coninue
            x_vals.append(x_vals[i] + deltat*e_k_x)
            y_vals.append(y_vals[i] + deltat*e_k_y)
            v_x_vals.append(v_x_vals[i] + deltat*e_k_vx)
            v_y_vals.append(v_y_vals[i] + deltat*e_k_vy)
            t_vals.append(t_vals[i] + deltat)
        else:   #if collides, stop iterations and return information
            crash_vel = np.sqrt(v_x**2+v_y**2)/1e3      #km/s
            crash_time = t_vals[-1]/3600                #hours
            if r_earth < R_E:   #collided with earth
                print('\n\nThe rocket crashed into the Earth, terminating simulation prematurely.')
                print('impact velocity = {:.3f} km/s| impact time = {:.3f} hours'.format(crash_vel,crash_time))
                earth_ticker += 1
            elif r_moon < R_M:  #collided with moon
                print('\n\nThe rocket crashed into the Moon, terminating simulation prematurely.')
                print('impact velocity = {:.3f} km/s| impact time = {:.3f} hours'.format(crash_vel,crash_time))
                moon_ticker += 1
            break
        
    return x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename


def mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation=''):
    '''Formats and analyses orbit statistics.'''

    min_moon_alt = np.min(r_moon_vals)-R_M                                  #calculates minimum distance to moon
    min_earth_alt = np.min(r_earth_vals)-R_E                                #calculates minimum distance to earth
    U,K,tot_energy = calc_energy(x_vals,y_vals,v_x_vals,v_y_vals,model)     #calculates energies
    
    theta_vals = []
    orbit_ticker = 0    #number of successful orbits
    for i in range(len(x_vals)):
        theta = (np.arctan2(y_vals[i],x_vals[i]))%(2*np.pi) #finds orbit angle from +ve x axis
        theta_vals.append(theta)
        if i != 0:  #skip first value, so not counting launch as a successful orbit
            if theta_vals[i-1] - theta_vals[i] >= 3:        #detects sudden large change i.e. passed through 2 pi        
                orbit_ticker += 1                           #add a successful orbit
            
    std_earth_r = np.std(r_earth_vals)/1e3      #km
    frac_change_U = np.std(U)/abs(np.mean(U))   #J
            
    if investigation == 'a-1-1' or investigation == 'a-1-2':    #if R-K4/Euler default circle or ellipse
        print('\nNumber of successful orbits: {}'.format(orbit_ticker)) #shows relevant info
        if orbit_ticker >= 1:
            print('Orbital period: {}'.format(t_vals[-1]/orbit_ticker))
        if earth_ticker == 0:
            print('''
Standard deviation of orbital radius: {:.3f}km
Standard deviation in total energy: {:.5f}% of mean total energy.'''.format(std_earth_r,frac_change_U))
    
    elif investigation == 'a-1-3' or investigation == 'a-2':    #if R-K4/Euler default slingshot or user input
        print('\nNumber of successful orbits: {}'.format(orbit_ticker)) #shows relevant info
        if orbit_ticker >= 1:      #if successful orbit made, shows relevant info
            print('Orbital period: {}'.format(t_vals[-1]/orbit_ticker))
            print('''
Closest Earth re-approach altitude: {:.3f}km
Radio transmission time at closest Earth re-approach altitude: {:.3f}s
Closest approach to moon surface: {:.3f}km'''.format(min_earth_alt/1e3,min_earth_alt/3e8,min_moon_alt/1e3,10))
        else:   #if no orbit successful, shows relevant info
            print('''
Rocket did not complete an orbit successfully.
Closest approach to moon surface: {:.3f}km'''.format(min_moon_alt/1e3,10))
    elif investigation == 'b':      #if trial and error, shows relevant info
        print('''\n\nMission report:
Number of Earth collisions: {}
Number of Moon collisions: {}
Number of stable trajectories: {}'''.format(earth_ticker,moon_ticker,num_tries-(earth_ticker+moon_ticker)))
        
    
def text_art():
    '''Makes a pretty ASCII art figure for visual reference of parameters.'''
    
    print('''\n\n   
                        ^     
                        | TO THE MOON  
                 
                      __ __
                   **       **
                *#             #*
              *#        y        #*      
             *#         ^         #*    
            *#          |          #*  
            ##           --> x     ##  
            *#                     #*
             *#       EARTH       #*    
              *#                 #*
                *#             #*
                   **       **
                      ^^ ^^ 
    \n''')
    

def linear_vals(variable, num_tries, val_range, x_0, y_0, v_x_0, v_y_0,deltat):
    '''Generates array of evenly-spaced values based on trial_error_input().'''
    
    if variable == '1':     #varying initial x position
        range_vals = np.linspace(x_0,x_0+val_range,num_tries)
    elif variable == '2':   #varying initial y position
        range_vals = np.linspace(y_0,y_0+val_range,num_tries)
    elif variable == '3':   #varying initial x velocity
        range_vals = np.linspace(v_x_0,v_x_0+val_range,num_tries)
    elif variable == '4':   #varying initial y velocity
        range_vals = np.linspace(v_y_0,v_y_0+val_range,num_tries)
    elif variable == '5':   #varying timestep
        range_vals = np.linspace(deltat,deltat+val_range,num_tries)
    
    return range_vals

    
# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================

def calc_energy(x_vals,y_vals,v_x_vals,v_y_vals,model):
    '''Calculates the kinetic, potential, and total energy at each iteration step
    in the orbit and analyses.'''
    
    indices = len(x_vals)
    U = np.zeros(indices)
    K = np.zeros(indices)
    
    for i in range(indices):
        x = x_vals[i]       #assign current variables
        y = y_vals[i]
        v_x = v_x_vals[i]
        v_y = v_y_vals[i]
        
        if model == False:    #if not including moon
            den_e = np.sqrt(x**2 + y**2)        #earth PE denominator
            U[i] = -GRAV * (M_E * M_R)/den_e    #PE
        elif model == True:  #if including moon
            den_e = np.sqrt(x**2 + y**2)                    #earth PE denominator
            den_m = np.sqrt(x**2 + (y-D_M)**2)              #moon PE denominator
            U[i] = -GRAV * M_R * (M_E/den_e + M_M/den_m)    #PE
        K[i] = 0.5 * M_R * (v_x**2+v_y**2)   #kinetic energy of rocket
    tot_energy = U+K                         #total energy is sum of potential and kinetic
    
    return U,K,tot_energy
    

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
        
def initiate_orbit_plot(arg):
    '''Initiates plots for showing orbital trajectories.'''
    
    formatter = EngFormatter(unit='m')    #modifies numbers with too many zeros
    
    ax = plt.subplot(arg)   #arg changes depending on whether plotting multiple in same figure
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    earth_pic = plt.Circle((0, 0), R_E, color='b')      #diagram of earth
    moon_pic = plt.Circle((0, D_M), R_M, color='grey')  #diagram of moon
    ax.add_artist(earth_pic)
    ax.add_artist(moon_pic)
    
    return ax  


def plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,xlabel,ylabel):
    '''Plots the orbit trajectory.'''
    
    plt.figure()

    ax = initiate_orbit_plot(111)
    ax.plot(x_vals,y_vals,'r')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename,bbox_inches='tight')
    print('\n\n{} complete'.format(filename))


def compare_plot(x_vals,y_vals,e_x_vals,e_y_vals,runtime,deltat):
    '''Plots orbits generated by RK4 and Euler methods side by side.'''

    plt.figure()
    ax=[]   #initialise empty axis
    ax.append(initiate_orbit_plot(121)) #for diagrams side by side
    ax.append(initiate_orbit_plot(122))
    
    ax[0].plot(x_vals,y_vals,'r')
    ax[1].plot(e_x_vals,e_y_vals,'r')
    ax[0].set_title('RK4 Method | step size = {}s'.format(deltat))     #title orbits
    ax[1].set_title('Euler Method | step size = {}s'.format(deltat))

    filename = 'RK4-Euler_Comparison.pdf'
    plt.savefig(filename,bbox_inches='tight')
    
    
def plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename):
    '''Plots the kinetic, potential, and total energy as a function of time
    throughout the orbit.'''
    
    U,K,tot_energy = calc_energy(x_vals,y_vals,v_x_vals,v_y_vals,model)   #calculate energies
    
    plt.figure()
    plt.style.use('default')
    formatter_x = EngFormatter(unit='s')
    formatter_y = EngFormatter(unit='J')    #modifies numbers with too many zeros
    ax_energy = []

    ax_energy.append(plt.subplot(211))
    ax_energy[0].xaxis.set_major_formatter(formatter_x)
    ax_energy[0].yaxis.set_major_formatter(formatter_y)
    plt.plot(t_vals,U, 'r',label='Potential Energy')    
    plt.ylabel('Potential Energy')
    ax_energy.append(ax_energy[0].twinx())  #plots potential and kinetic energies on same graph
    ax_energy[1].xaxis.set_major_formatter(formatter_x)
    ax_energy[1].yaxis.set_major_formatter(formatter_y)
    plt.plot(t_vals,K, 'b',label='Kinetic Energy')
    plt.ylabel('Kinetic Energy')
    ax_energy[0].legend(loc = 1)
    ax_energy[1].legend(loc = 2)

    ax_energy.append(plt.subplot(212))      #plots total energy beneath other graph
    ax_energy[2].xaxis.set_major_formatter(formatter_x)
    ax_energy[2].yaxis.set_major_formatter(formatter_y)
    
    filename = filename[0:-4]+'energy.pdf'
    plt.plot(t_vals,tot_energy, 'g')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy')
    plt.savefig(filename,bbox_inches='tight')


# =============================================================================
# 2ND LEVEL MENU FUNCTIONS
# =============================================================================
    
    
def default_menu(model):
    '''Menu function for choosing between pre-coded RK4 scenarios.'''
    
    Def_Input = '0'
    print('For each scenario, a single successful orbit is completed.')
    while Def_Input != 'q':
        def_question = '''
        Characteristic orbits menu (RK4):
        Enter a choice - "1" (circular orbit),
                         "2" (elliptical orbit),
                         "3" (moon hourglass orbit)
                          - or "q" to quit defaults menu: '''

        Def_Input = input(def_question).lower()    #ensures both 'q' and 'Q' accepted
        if Def_Input == '1':        #circular orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_circular_orbit.pdf', 5550, 10, R_E+408e3, 0, 0, 7.66e3)
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-1')
        elif Def_Input == '2':      #elliptical orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_elliptical_orbit.pdf', 34240, 10, R_E+408e3, 0, 0, 10e3)
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-2')
        elif Def_Input == '3':      #hourglass orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_slingshot_orbit.pdf',814050, 100, 0, -7.1e6, 10.4801e3, 0)
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-3')
        elif Def_Input != 'q':      #must enter '1', '2', or '3'
            print('Choice not valid.')
            
    
def default_euler_menu(model):
    '''Menu function for choosing between pre-coded Euler scenarios.'''
    
    Def_Input = '0'
    while Def_Input != 'q':
        def_question = '''
        For each scenario, a single successful orbit is completed.
        Characteristic orbits menu (Euler):
        Enter a choice - "1" (circular orbit),
                         "2" (elliptical orbit),
                         "3" (moon slingshot orbit)
                          - or "q" to quit defaults menu: '''

        Def_Input = input(def_question).lower()    #ensures both 'q' and 'Q' accepted
        if Def_Input == '1':        #circular orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = euler_calc(model, 6000, 100, R_E+408e3, 0, 0, 7.66e3, 'Euler_circular_orbit.pdf')
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-1')
        elif Def_Input == '2':      #elliptical orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = euler_calc(model, 35000, 100, R_E+408e3, 0, 0, 10e3, 'Euler_elliptical_orbit.pdf')
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-2')
        elif Def_Input == '3':      #hourglass orbit functions
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = euler_calc(model,814050, 100, 0, -7.1e6, 10.4801e3, 0, 'Euler_slingshot_orbit.pdf')
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-1-3')
        elif Def_Input != 'q':      #must enter '1', '2', or '3'
            print('Choice not valid.')
    
    
# =============================================================================
# 1ST LEVEL MENU FUNCTIONS
# =============================================================================
    
def RK4_menu():
    '''Menu function for RK4 orbit method.'''
    
    RK4_Input = '0'
    model = yesno('                 [y/n] Take moon into account? ')
    while RK4_Input != 'q':
        rk4_question = '''
    RK4 menu:
    Enter a choice - "1" (default values),
                     "2" (user input values)
                      - or "q" to quit RK4 menu: '''

        RK4_Input = input(rk4_question).lower()    #ensures both 'q' and 'Q' accepted
        if RK4_Input == '1':    #goes to menu of pre-coded scenarios
            default_menu(model)
        elif RK4_Input == '2':  #user input scenarios functions
            runtime, deltat = simp_inputting()
            text_art()          #visual reference for parameters
            x_0, y_0, v_x_0, v_y_0 = initial_values_input()
            filename = 'RK4_inputted_orbit.pdf'
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x','y')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-2')
        elif RK4_Input != 'q':  #must enter '1' or '2'
            print('Choice not valid.')
            
            
def trial_error():
    '''Generates array of plots for different initial values.'''
    
    runtime, deltat = simp_inputting()                  #gets runtime and time step
    variable, model = change_var()
    x_0, y_0, v_x_0, v_y_0 = initial_values_input()     #gets initial values
    num_tries, val_range = trial_error_input(variable)  #gets number of trials, and range of trials
    range_vals = linear_vals(variable, num_tries, val_range, x_0, y_0, v_x_0, v_y_0,deltat) #generates range of values
    
    plt.figure()
    ax = initiate_orbit_plot(111)   #sets up plot
    
    for i in range(num_tries):
        style_types = ['r-','r:','m-','m:','b-','b:','k-','k:','c-','c:']   #sets up alternating line stypes
        style = style_types[i%10]
        if variable == '1':         #varying initial x position
            x_0 = range_vals[i]     #generates evenly-spaced trial values
            filename = 'RK4-var-x:({:.0f})-({:.0f})km.pdf'.format(range_vals[0],range_vals[-1])   #generates file name to save to
            print('\n\nTrial {}/{} | initial x: {:.0f}m'.format(i+1,num_tries,x_0))               #trial information
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)    #generate RK4 orbit
            ax.plot(x_vals,y_vals,style,label='{:.0f}'.format(x_0/1e3)) #plot orbital trajectory
            leg_title = 'Initial {} value (km)'.format('x_0')           #legend title          
        elif variable == '2':       #varying initial y position
            filename = 'RK4-var-y:({:.0f})-({:.0f})km.pdf'.format(range_vals[0],range_vals[-1])
            y_0 = range_vals[i]
            print('\n\nTrial {}/{} | initial y: {:.0f}m'.format(i+1,num_tries,y_0))
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)
            ax.plot(x_vals,y_vals,style,label='{:.0f}'.format(y_0/1e3)) #plot orbital trajectory
            leg_title = 'Initial {} value (km)'.format('y_0')           #legend title
        elif variable == '3':       #varying initial x velocity
            filename = 'RK4-var-v_x:({:.0f})-({:.0f})kms.pdf'.format(range_vals[0],range_vals[-1])
            v_x_0 = range_vals[i]
            print('\n\nTrial {}/{} | initial v_x: {:.0f}m/s.pdf'.format(i+1,num_tries,v_x_0))
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)
            ax.plot(x_vals,y_vals,style,label='{:.3f}'.format(v_x_0/1e3))   #plot orbital trajectory
            leg_title = 'Initial {} value (km/s)'.format('v_x_0')           #legend title
        elif variable == '4':       #varying initial y velocity
            filename = 'RK4-var-v_y:({:.0f})-({:.0f})kms.pdf'.format(range_vals[0],range_vals[-1])
            v_y_0 = range_vals[i]
            print('\n\nTrial {}/{} | initial v_y: {:.0f}m/s'.format(i+1,num_tries,v_y_0))
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)
            ax.plot(x_vals,y_vals,style,label='{:.3f}'.format(v_y_0/1e3))   #plot orbital trajectory
            leg_title = 'Initial {} value (km/s)'.format('v_y_0')           #legend title
        elif variable == '5':       #varying time step
            filename = 'comparison-stepsize:({:.0f})-({:.0f})s.pdf'.format(range_vals[0],range_vals[-1])
            deltat = range_vals[i]
            if deltat > runtime:    #if trying a step size greater than the run time, warn
                print('Time step out of chosen simulation runtime ({}s). No graph plotted.'.format(runtime))
            print('\n\nTrial {}/{} | step size: {:.0f}s'.format(i+1,num_tries,deltat))
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,filename,runtime, deltat, x_0, y_0, v_x_0, v_y_0)
            ax.plot(x_vals,y_vals,style,label='{:.0f}'.format(deltat))      #plot orbital trajectory
            leg_title = 'Initial {} value (s)'.format('time step')          #legend title
            
    #graph formatting
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.legend(title = leg_title)
    ax.get_legend().get_title()
    plt.savefig(filename,bbox_inches='tight')
    print('\n\n{} complete.'.format(filename))  #show file completed
            

def compare_menu():
    '''Menu function for comparing results of pre-coded scenarios for Euler 
    and RK4 methods.'''
    
    Comp_Input = '0'
    model = yesno('                 [y/n] Take moon into account? ')
    while Comp_Input != 'q':
        comp_question = '''
    R-K4 vs Euler comparison menu:
    Enter a choice - "1" (circular orbit)
                     "2" (elliptical orbit),
                     "3" (moon slingshot orbit),
                      - or "q" to quit comparison menu: '''

        Comp_Input = input(comp_question).lower()    #ensures both 'q' and 'Q' accepted
        if Comp_Input == '1':        #circular orbit functions
            runtime, deltat = simp_inputting()
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_circular_orbit.pdf',runtime, deltat, R_E+408e3, 0, 0, 7.66e3)
            e_x_vals,e_y_vals,e_v_x_vals,e_v_y_vals,e_t_vals,e_earth_ticker,e_moon_ticker,e_r_earth_vals,e_r_moon_vals,e_filename = euler_calc(model,runtime, deltat, R_E+408e3, 0, 0, 7.66e3, 'Euler_circular_orbit.pdf')
            compare_plot(x_vals,y_vals,e_x_vals,e_y_vals,runtime,deltat)
        elif Comp_Input == '2':     #elliptical orbit functions
            runtime, deltat = simp_inputting()
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_elliptical_orbit.pdf',100000, 100, R_E+408e3, 0, 0, 10e3)
            e_x_vals,e_y_vals,e_v_x_vals,e_v_y_vals,e_t_vals,e_earth_ticker,e_moon_ticker,e_r_earth_vals,e_r_moon_vals,e_filename = euler_calc(model,100000, 100, R_E+408e3, 0, 0, 10e3, 'Euler_elliptical_orbit.pdf')
            compare_plot(x_vals,y_vals,e_x_vals,e_y_vals,runtime,deltat)
        elif Comp_Input == '3':     #slighshot orbit functions
            runtime, deltat = simp_inputting()
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = RK4_orbit(model,'RK4_slingshot_orbit.pdf',814050, 100, 0, -7.1e6, 10.4801e3, 0)
            e_x_vals,e_y_vals,e_v_x_vals,e_v_y_vals,e_t_vals,e_earth_ticker,e_moon_ticker,e_r_earth_vals,e_r_moon_vals,e_filename = euler_calc(model,815000, 100, 0, -7.1e6, 10.4801e3, 0, 'RK4_slingshot_orbit.pdf')
            compare_plot(x_vals,y_vals,e_x_vals,e_y_vals,runtime,deltat)
        elif Comp_Input != 'q':      #must enter '1', '2', or '3'
            print('Choice not valid.')
            
            
def euler_menu():
    '''Menu function for Euler orbit method.'''
    
    Euler_Input = '0'
    model = yesno('                 [y/n] Take moon into account? ')
    while Euler_Input != 'q':
        euler_question = '''
    Euler menu:
    Enter a choice - "1" (default values),
                     "2" (user input values)
                      - or "q" to quit Euler menu: '''

        Euler_Input = input(euler_question).lower()    #ensures both 'q' and 'Q' accepted
        if Euler_Input == '1':      #pre-coded scenarios
            default_euler_menu(model)
        elif Euler_Input == '2':    #user input scenarios
            runtime, deltat = simp_inputting()
            text_art()              #visual reference for choosing parameters
            x_0, y_0, v_x_0, v_y_0 = initial_values_input()
            x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename = euler_calc(model,runtime, deltat, x_0, y_0, v_x_0, v_y_0,'Euler_inputted_orbit.pdf')
            plot_orbit(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,filename,'x coordinate','y coordinate')
            plot_energy(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,model,filename)
            mission_report(x_vals,y_vals,v_x_vals,v_y_vals,t_vals,earth_ticker,moon_ticker,r_earth_vals,r_moon_vals,filename,model,num_tries='',investigation='a-2')
        elif Euler_Input != 'q':  #must enter '1' or '2'
            print('Choice not valid.')
        

# =============================================================================
# INITIALISE FUNCTIONS
# =============================================================================

greet()

folder_ask = yesno('''[y/n] Would you like graphs written to .pdf files?
            (will also be displayed once script exited): ''')

if folder_ask == True:
    myfolder()  #creates new folder if necessary for storage of outputted graph .png files

MyInput = '0'
while MyInput != 'q':
    mainquestion = '''
Main menu:
Enter a choice - "a" (R-K4 orbit),
                 "b" (R-K4 repeats with varying initial variables), 
                 "c" (integration method (R-K4 vs Euler) comparison plots),
                 "d" (Euler orbit)
                  - or "q" to quit program: '''
                  
    MyInput = input(mainquestion).lower()
    
    if MyInput == 'a':      #R-K4 orbits
        RK4_menu()
    elif MyInput == 'b':    #vary initial parameters
        trial_error()
    elif MyInput == 'c':    #compare Euler and R-K4
        compare_menu()
    elif MyInput == 'd':    #Euler orbits
        euler_menu()
    elif MyInput != 'q':    #incorrect response
        print('Choice not valid.')
        
print('\nYou have chosen to finish - goodbye, and have a nice day!')
plt.show()      #shows all figures at close of program
