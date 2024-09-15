import numpy as np
import types
from types import SimpleNamespace
import numpy.random as rnd
import necessary_func as nf
import SW_algorithm as sw
import bootstrapping_and_observables as bo
import metropolis_algorithm as ma
import necessary_func as nf


#############################################################################################################


def sw_simulation(params):
    """
    The function is used to simulate the Swendsen-Wang algorithm.
    """
    
    spins = nf.spin_config(params)
    xy_coordinates, spin_numbers = nf.coordinat_latice(params)
    total_spins = (params.L)**2
    
    Temperature, energy, chi, c_v, magnetisation, energy_mc, magnetisation_mc, chi_mc, cv_mc = nf.initializing_array(params)
    
    
    for p, temp in enumerate(range(params.T_len)):
        ran_energy = nf.system_energy(params, xy_coordinates, spins, spin_numbers)
        
        for i, t in enumerate(range(params.mc_steps)):
            cluster_islands, spins, cluster_is_flipped = sw.SW_algorithm(params, xy_coordinates, spin_numbers, spins)
            energy_mc[i] = nf.system_energy(params, xy_coordinates, spins, spin_numbers)
            magnetisation_mc[i] = np.mean(spins)
                
                
        energy[p] = np.mean(energy_mc[-params.eq_steps:]) /params.total_spins
        m_squared = (magnetisation_mc)**2

        btstrp_sequence = bo.bootstrapping_random(params)
        magnetisation[p] = bo.mag_calc(params, m_squared, btstrp_sequence)
        chi[p] = bo.chi_calc(params, abs(magnetisation_mc), btstrp_sequence)
        c_v[p] = bo.c_v_calc(params, energy_mc, btstrp_sequence)
        
        Temperature[p] = params.T 
       
        params.T = params.T + params.dT
    
    
    results = SimpleNamespace(temperature = Temperature,
                              chi = chi,
                              energy = energy,
                              magnetisation = magnetisation,
                              c_v = c_v,
                              spins = spins,
                              coordinates = xy_coordinates,
                              cluster_islands = cluster_islands
                             )
        
    return results
    
    
#############################################################################################################
    
    
def ising_flipping(params):
    """
    The function is used to simulate the ising model through single spin flip, and
    doing N flips in one MC steps.
    """
    
    spins = nf.spin_config(params)
    xy_coordinates, spin_numbers = nf.coordinat_latice(params)
    total_spins =(params.L) **2
    
    Temperature, energy, chi, c_v, magnetisation, energy_mc, magnetisation_mc, chi_mc, cv_mc = nf.initializing_array(params)
    
    for p, temp in enumerate(range(params.T_len)):
        
        ran_energy = nf.system_energy(params, xy_coordinates, spins, spin_numbers)
        
        MCS = params.L 
        MCS = MCS **2
        time_steps = MCS * params.mc_steps
        
        for i, tom in enumerate(range(time_steps)):
            
            if tom % MCS == 0:
            
                energy_mc[int(i/MCS)] = ran_energy 
                magnetisation_mc[int(i/MCS)] = np.mean(spins)
                spins, ran_energy = ma.metropolis_algorithm(params, xy_coordinates, spins, ran_energy)
                
                
        energy[p] = np.mean(energy_mc[-params.eq_steps:]) /params.total_spins
        m_squared = (magnetisation_mc)**2

        btstrp_sequence = bo.bootstrapping_random(params)
        magnetisation[p] = bo.mag_calc(params, m_squared, btstrp_sequence)
        chi[p] = bo.chi_calc(params, abs(magnetisation_mc), btstrp_sequence)
        c_v[p] = bo.c_v_calc(params, energy_mc, btstrp_sequence)
        
        Temperature[p] = params.T 
       
        params.T = params.T + params.dT
    
    
    results = SimpleNamespace(temperature = Temperature,
                              chi = chi,
                              energy = energy,
                              magnetisation = magnetisation,
                              c_v = c_v,
                              spins = spins,
                              coordinates = xy_coordinates,
                             )
        
    return results
    
#############################################################################################################

    
    


