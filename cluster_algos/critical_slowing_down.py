import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit



"""
The module is helpful for fitting the functions to get the critical exponents, and plots the graphs of 
    autocorrelation fucntions. 
"""


########################################################################################

def load_data(data_dir, quantity):
    
    data = np.load(data_dir)
    x_data = data['temperature'].reshape(np.shape(data['temperature'])[0])
    y_data = data[quantity][:, 0]
    y_err = data[quantity][:, 1]
    
    return x_data, y_data, y_err


########################################################################################

def func_chi(tau, factor, a):
    
    '''
    This is the trial fitting function for the critical exponwent of magnetic susceptibility.  
        
    ''' 
    func = factor*tau**a
    return func

########################################################################################

def func_cv(tau, factor):
    '''
    This is the trial fitting function for the critical exponwent of specific heat.  
        
    ''' 
    
    func = factor*np.log(tau)
    return func


########################################################################################

def f_magnetisation(tau, factor, a):
    
    '''
    This is the trial fitting function for critical exponent of magnetisation.   
    
    ''' 
    
    func = factor*tau**a
    return func                  


########################################################################################


def f_z(L, x, c):
    '''
    This is the trial fitting function for the critical dynamical exponent.  
    
    '''
    
    func = L**(x) + c
    return func

########################################################################################

def fit_funct_z(f_z, L, Y, err, bounds):
    
    popt, pcov = curve_fit(f_z, L, Y, sigma = err, bounds = bounds)
    fit_err = np.sqrt(np.diag(pcov))
    
    return popt, fit_err


########################################################################################


def f_sim(x, a, b, c):
    
    linear = a * x**(b) + c
    return linear

##########################################################################################
    
    
def fit_funct_sim(f_sim, X, Y):
    
    popt, pcov = curve_fit(f_sim, X, Y)
    fit_err = np.sqrt(np.diag(pcov))
    return popt, fit_err


#########################################################################################

#########################################################################################

###Plotting the functions


def grid_plot(params, results, fig_dir, identifier, save):
    
    '''
    This functions generates the plots that is required to observe cluster, or even equilibrium.
    '''
    
    figure_directory = fig_dir
    
    x = results.coordinates[0]
    y = results.coordinates[1]
    S = results.spins
    
    plt.figure(figsize = (12, 8))    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=20)
    
    
    image = plt.imshow(S, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.plasma)
    plt.clim(-1,1)
    plt.xlabel('y')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('x')
    
    if save:
        plt.savefig(figure_directory + identifier + '_' +'grid_spins.pdf')
    plt.close()
    
    
#########################################################################################

algorithm = 'SW'

def visualize_islands(params, results, fig_dir, identifier, save):
    '''
    This function gives the plot for the clusters in the coordintes space. 
    
    Parameters
    ----------
    params : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results  
    fig_dir : str
        directory where figures should be stored
    identifier : str
        identifier of the data set, is used in filename 
    save : bool
        determines if files should be saved
        
    '''
    
    algorithm = 'SW'
    
    if algorithm == 'SW':

        figure_directory = fig_dir

        islands = results.cluster_islands

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=18)

        m_size = 47000/(params.L*params.L)
        # Visualize spin islands
        x_coord = []
        y_coord = []

        for i in islands:
            x_temp = []
            y_temp = []
            for x, y in i:
                x_temp.append(x)
                y_temp.append(y)

            x_coord.append(x_temp)     
            y_coord.append(y_temp)  
        plt.figure(figsize=(12, 8))
        for i, x in enumerate(x_coord):
            y = y_coord[i]
            plt.scatter(y, x, s=m_size, marker="s")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-0.5, params.L - 0.5])
        plt.ylim([params.L - 0.5, -0.5])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('y')
        plt.ylabel('x')

        if save:
            plt.savefig(figure_directory + identifier + '_' + 'clusters.pdf')
        plt.close()
        
        
        
#########################################################################################


def plot_func(params, results, fig_dir, identifier, save):
    '''
    This function plots all the required quantities, m, c_v, chi.
    
    Parameters
    ----------
    params : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulational results  
    fig_dir : str
        directory where figures should be stored
    identifier : str
        identifier of the data set, is used in filename 
    save : bool
        determines if files should be saved
    '''

    algorithm = 'SW'
    figure_directory = fig_dir
    fig_name = figure_directory + identifier + '_' + algorithm + str(params.eq_steps) 
    
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    x = results.temperature
    y = results.c_v[:,0]
    y_err = results.c_v[:,1]
    
    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$\mathrm{C_v}$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_cv.pdf')
    plt.close()

    x = results.temperature
    y = results.chi[:,0]
    y_err = results.chi[:,1]
    
    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$\chi$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_chi.pdf')
    plt.close()
    
    x = results.temperature
    y = results.magnetisation[:,0]
    y_err = results.magnetisation[:,1]
    plt.plot(x, y, 'x', markersize=6)
    #plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$ m^{2} $', fontsize=18)    
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_m_sq.pdf')
    plt.close()
    
    
    if save:
        print('Figures are saved to: ' + figure_directory)
        
        
#########################################################################################


def crit_temp(data_dir):
    '''
    Gives the transition/critical temperature from the measured specific
    heat in terms of k_B*T/J.
    
    '''
    
    data = np.load(data_dir)
    T = data['temperature'].reshape(np.shape(data['temperature'])[0])
    c_v = data['c_v'][:,0]
    T_c = T[c_v==max(c_v)]
    
    return T_c

######################################################################################

def fit_function(data_dir, quantity, fit_range, plotYN, LOG):
    '''
    Gives the best fit to quantity with non-linear least squares.
    '''
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    # Import needed functions from data_processing.py
    from critical_slowing_down import crit_temp, load_data
       
    # Loading data and general functions (and generate transition temperature (T_c))
    from scipy.optimize import curve_fit
    
    data = np.load(data_dir)
    xdata = data['temperature'].reshape(np.shape(data['temperature'])[0])
    ydata = data[quantity][:,0]
    y_err = data[quantity][:,1]
    
    T_c = crit_temp(data_dir)
    
    # Fitting
    ## Selecting data for fit
    indices = np.where(((xdata-T_c)>fit_range[0]) & ((xdata-T_c)<fit_range[1]))
    xdata_fit = xdata[indices] - T_c
    ydata_fit = ydata[indices]
    y_err_fit = y_err[indices]

    ## Select fitting model
    if quantity == 'magnetisation':
        from critical_slowing_down import f_magnetisation as f
        plt.ylabel(r'$\langle \mathrm{m^2} \rangle$', fontsize=18)
    if quantity == 'c_v':
        from critical_slowing_down import func_cv as f
        plt.ylabel('$\mathrm{C_v}$', fontsize=18)
    if quantity == 'chi':
        from critical_slowing_down import func_chi as f
        plt.ylabel('$\mathrm{\chi}$', fontsize=18)
    
    if quantity == 'magnetisation':
        ## Actual fitting
        popt, pcov = curve_fit(f, abs(xdata_fit), ydata_fit, sigma=y_err_fit)
        fit_err = np.sqrt(np.diag(pcov))

        # Generate fit plot data
        xdata_fit_plot = np.linspace(xdata_fit[0], xdata_fit[-1], 1000)
        ydata_fit_plot = f(abs(xdata_fit_plot), *popt)
                
        # Plotting original function and fit
        if plotYN:

            # Actual plotting
            if LOG:
                # Actual plotting
                plt.loglog(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.loglog(abs(xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.loglog(abs(xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit -')
                plt.xlabel('$\mathrm{k_B |T-T_c|/J}$')
                plt.grid()
            else:
                plt.plot(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.plot((xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.plot((xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit -')
                plt.xlabel('$\mathrm{k_B (T-T_c)/J}$')
                
            plt.tight_layout()
            #plt.show()

        return popt, fit_err, xdata_fit_plot, ydata_fit_plot
        
    else:
        ## Actual fitting
        popt, pcov = curve_fit(f, abs(xdata_fit), ydata_fit, sigma=y_err_fit)
        fit_err = np.sqrt(np.diag(pcov))

        # Generate fit plot data
        xdata_fit_plot = np.linspace(xdata_fit[0], xdata_fit[-1], 1000)
        ydata_fit_plot = f(abs(xdata_fit_plot), *popt)
        
        # Plotting original function and fit
        if plotYN:
            if LOG:
                # Actual plotting
                plt.loglog(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.loglog(abs(xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.loglog(abs(xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit +')
                plt.xlabel('$\mathrm{k_B |T-T_c|/J}$')
                plt.grid()
                
            else:
                plt.plot(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.plot((xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.plot((xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit +')
                plt.xlabel('$\mathrm{k_B (T-T_c)/J}$')
            
            plt.tight_layout()
            #plt.show()

        return popt, fit_err, xdata_fit_plot, ydata_fit_plot


#########################################################################################
        
def save_data(params, results, data_dir, identifier):
    
    '''
    Saves most imported data to a npz file. This file
    can contain multiple numpy arrays.     
    '''
    algorithm = "SW"
    
    data_name = data_dir + 'saved_data_' + identifier + '_' + algorithm + '_' + str(params.eq_steps)
    np.savez(data_name, temperature = results.temperature,
             c_v = results.c_v, chi = results.chi, magnetisation = results.magnetisation,
            sim_time = results.sim_time)
    print('Data is saved to: ' + data_dir)
    
    
##########################################################################################
