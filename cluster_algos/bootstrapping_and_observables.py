import numpy as np
import random
import necessary_func as nf
import metropolis_algorithm as ma


#############################################################################################################

def bootstrapping_random(params):
    
    """
    The function generates the random number which are used for the sampling estimates.
    
    Args:
        params (_type_): _SimpleNamespace which contains all the initial parameter to run the simulation_
        energy_system (_float_)
        xy_coordinates (_2D_array_): _containing the coordinates of each point on the lattice_
        spins (_type_): _constains array of spins fr each lattice site_
    """
    
    """

    Returns:
        _matrix_: _which conatins the different randomly generated numbers which we will use for
        sampling estimates to demand higher accuracy in the simulation_
    """
    
    number = params.eq_steps - 1
    bootstrap_trails = params.bootstrap_trails
    
    matrix = np.random.random(number * bootstrap_trails).reshape(bootstrap_trails, number)
    matrix = matrix * number
    matrix = np.round(matrix)
    
    return matrix.astype(int)
    
    
#############################################################################################################
    
    
def mag_calc(params, m_squared, btstrp_sequence):
    
    """
    The function calculates the magnetisation of the system.
    
    Args:
        spins : 2D array (L, L)
        btstrp_seq: 2D array (trials, eq_steps)
        contains random sequence of samples that can be taken from the data, 
        for the ammount of bootstrap trials that are required
    """
    
    """
    Returns:
        _float_: _ average magnitization, floating number
        standard deviation of magnitisation, 
        m_sigma : float, standard deviation
    """
    
    eq = m_squared[-params.eq_steps:]
    mag = np.zeros((params.bootstrap_trails, 1), dtype=float)
    
    for j in range(params.bootstrap_trails):        
        sample = eq[btstrp_sequence[j]]        
        mag[j] = np.mean(sample)

    m_average = np.mean(mag)
    m_sigma = np.std(mag)
    
    return m_average, m_sigma 


#############################################################################################################


def chi_calc(params, magnetisation, btstrp_sequence):
    """
    This calculates the susceptibility of the lattice system.
    
    Args:
    magnetisation: 1D array (1, eq_steps)  
    btstrp_sequence: 2D array (trials, eq_steps) generated from the bootstrapping_random function.
        
    it, returns the
    -------
    chi_average : float
        which includes the average susceptibility of the system
    chi_sigma : float
        which includes the sigma of the specific susceptibility of the system  
        
    """
    
    eq = magnetisation[-params.eq_steps:]
    
    chi_sm = np.zeros((params.bootstrap_trails, 1), dtype=float)
    for j in range(params.bootstrap_trails):        
        sample = eq[btstrp_sequence[j]]        
        chi_sm[j] = (params.total_spins)*np.var(sample)/((params.T)*(params.kb))

    chi_average = np.mean(chi_sm)
    chi_sigma = np.std(chi_sm)
    
    return chi_average, chi_sigma


#############################################################################################################


def c_v_calc(params, energy_mc, btstrp_sequence):
    
    """
    The function calculates the specific heat of the system.
    Also, it returns the average specific heat and the standard deviation of the specific heat.
    """
    
    eq = energy_mc[-params.eq_steps:]
    c_v_sm = np.zeros((params.bootstrap_trails, 1), dtype=float)
    for j in range(params.bootstrap_trails):        
        sample = eq[btstrp_sequence[j]]
        c_v_sm[j] = np.var(sample)/((params.total_spins)*(params.T**2)*(params.kb))

    c_v_average = np.mean(c_v_sm)
    c_v_sigma = np.std(c_v_sm)

    return c_v_average, c_v_sigma


#############################################################################################################
