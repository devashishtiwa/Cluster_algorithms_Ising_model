import numpy as np
import random


###############################################################################################################

def spin_config(params):
    
    """
    The code initializes the system with a specific configuration.
    """

    """
    It takes the parameters from the SimpleNamespace object, whose arguments can be 
    stored using the types initially in the system. 
    
    The function actually helps us to create the iniial configuration of various kinds,
    either we want all spins up or all down, or maybe probable, like 50 percent up and
    50 percent down, depending upon our requirements. One need to pass such argument later on, what
    kind of initial configuration one actually want.

    """
    
    Lx = params.L
    Ly = params.L
    
    spins = np.zeros(Lx*Ly)
    
    if params.initial_config == 'random':
        spins = np.random.rand(params.L, params.L)
        spins[spins >= 0.5] =  1
        spins[spins <  0.5] = -1
        
    elif params.initial_config == 'up':
        spins = np.ones([params.L,params.L], dtype= int)
        
    elif params.initial_config == 'down':
        spins = -1*np.ones([params.L,params.L], dtype= int)
        
    else:
        print("Give valid initial configuration, which can either be up or down or random")
        
    return spins


###############################################################################################################

def coordinat_latice(params):
    """
    The code is about the x and y coordinates of the lattice and the unique numbers given to each lattice point.

    Args:
        params: _uses the parameter of length of the lattice_

    Returns:
        2, 1D-array : coordinates in the lattice and numbering of the spins which are unique
    """
    
    x, y = [range(params.L), range(params.L)]
    xy_coordinates = np.meshgrid(x, y)
    xy_coordinates = np.reshape(xy_coordinates, (2,-1))
    spin_numbers = range(params.total_spins)

    return xy_coordinates, spin_numbers


###############################################################################################################

def initializing_array(params):
    
    """
    This initializes all the array, and also the 
    corresponding quantities that are observables and are recorded in 
    the simulation, to identify the phase transition.
    """

    Temp = np.zeros([params.T_len, 1])
    energy = np.zeros([params.T_len, 1])
    magnetisation = np.zeros([params.T_len, 2])
    chi = np.zeros([params.T_len, 2])
    c_v = np.zeros([params.T_len, 2])    
    energy_mc = np.zeros([params.mc_steps, 1])
    magnetisation_mc = np.zeros([params.mc_steps, 1])
    cv_mc = np.zeros([params.mc_steps, 1])
    chi_mc = np.zeros([params.mc_steps, 1])
    

    return Temp, energy, chi, c_v, magnetisation, energy_mc, magnetisation_mc, chi_mc, cv_mc


###############################################################################################################

def onsite_energy(params, x_coordinates, y_coordinates, spins):
    
    """
    The function helps us to calculate the onsite energy of each spin lattice. 
    
    The function requires the following parameters, 
    namely, 
    x_coordinates : float variable
    y_coordinates : float variable
    spins : This is a 2D array of dimension (L, L) containing all the spins in the lattice.
    
    The function returns the output the site energy of each lattice point, because of other four neighbours.
        
    """
    onsite_energy = 0
    
    x_neigbour = (x_coordinates + np.array([1, 0, -1, 0]))%(params.L)
    y_neigbour = (y_coordinates + np.array([0, -1, 0, 1]))%(params.L)
    
    for i in range(np.size(x_neigbour)):
        spin_center = spins[x_coordinates, y_coordinates]
        spin_value_neighbour = spins[x_neigbour[i], y_neigbour[i]]

        onsite_energy += -params.J*spin_center*spin_value_neighbour
        
    return onsite_energy 


#############################################################################################################


def system_energy(params, xy_coordinates, spins, spin_numbers):
    
    """
    The function helps us to calculate the total energy of the system.
    
    The function requires the following parameters, 
    namely, 
    xy_coordinates : float variable (2D array containing all the coordinates of the system)
    spins : This is a 2D array of dimension (L, L) containing all the spins in the lattice.
    spin_numbers : float variable
    
    The function returns the output the total energy of the system.
        
    """
    
    energy = 0
    
    for i in spin_numbers:
        x = xy_coordinates[0][i]
        y = xy_coordinates[1][i]
        energy += onsite_energy(params, x, y, spins)
        
    return energy/2


#############################################################################################################

