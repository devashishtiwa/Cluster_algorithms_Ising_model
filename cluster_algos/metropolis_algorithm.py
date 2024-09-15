import numpy as np
import random
import necessary_func as nf
import numpy.random as rnd


#############################################################################################################

def accepting_spin_flip(params, energy_diff):
    
    """_summary_
    This code actually tells about whether we will accept the spin flip or not.
    using metropolis probability criteria.
    """
    
    """"
    The function takes the argument, 
    params: which is the SimpleNamespace object, which contains the parameters of the system.
    energy_diff: which is the difference in the energy of the system after the spin flip.
    """

    """
    Returns the boolean value, whether to accept the spin flip or not.
    """
    
    return rnd.binomial(1, np.exp(-energy_diff/(params.kb*params.T))) == 1
    
#############################################################################################################
    
def metropolis_algorithm(params, xy_coordinates, spins, energy_system):
    
    """
    The code is about the metropolis algorithm, which is used to flip the spins in the system.
    
    Args:
        params (_type_): _uses the parameter of length of the lattice_
        spins (_array_): _contains all the spins in the system_
        xy_coordinates: 2D array conatining all the x and y coordinates of the lattice points.
        energy: energy of the system.
        
    Returns:
        _type_: spins after the flip
    """
    
    random_site_number = np.random.randint(0, params.total_spins)
    x_coordinates = xy_coordinates[0][random_site_number]
    y_coordinates = xy_coordinates[1][random_site_number]
    
    energy_before_flipping = nf.onsite_energy(params, x_coordinates, y_coordinates, spins)
    spins[x_coordinates, y_coordinates] = -1*spins[x_coordinates, y_coordinates]
    energy_after_flipping = nf.onsite_energy(params, x_coordinates, y_coordinates, spins)
    
    energy_diff = energy_after_flipping - energy_before_flipping
    
    if energy_diff <= 0 :
        energy_system += energy_diff
    else:
        if accepting_spin_flip(params, energy_diff):
            energy_system += energy_diff
        else:
            spins[x_coordinates, y_coordinates] = -1*spins[x_coordinates, y_coordinates]
            
    return spins, energy_system
    
#############################################################################################################

    
    
    
    