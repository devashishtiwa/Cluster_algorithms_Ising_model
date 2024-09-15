import numpy as np
import random
import numpy.random as rnd


#############################################################################################################


def Wolff_algorithm(params, xy_coordinates, spin_numbers, spins):
    
    """
    This function identifies all the possible cluster and choose the one which is largest of all and flip that. This isthe most crucial part of wolff algorithm.
    
    This is almost same as the SW algorithm, just this function makes the difference where we have just one cluster, unlike group of clusters. 
    
    Returns:
        cluster island: containsing just the one clusters, 
        spins: after flipping with 1/2 probability (made sure by flippping_the_cluster array),
        cluster_is_flipped : which is a list which tells whether the cluster is flipped or not
    """
    
    cluster_island = []                                                    #Here this island just contains one cluster unlike SW Algorithm
    cluster_is_flipped = []
    n_visited = np.ones((params.L, params.L), dtype= bool)
    
    bonds = bonds_evalulation(params, spins)
    
    for i in spin_numbers:
        cluster = []
        flipping_the_cluster = 2*np.random.randint(2) - 1 
        x_coordinates = xy_coordinates[0][i]
        y_coordinates = xy_coordinates[1][i]
        cluster, spins = back_tracking(params, x_coordinates, y_coordinates, bonds, n_visited, cluster, spins, flipping_the_cluster)
    
        if cluster != []:
            cluster_island.append(cluster)
            cluster_is_flipped.append(flipping_the_cluster)                
    
    #now choose one cluster which is bigger of all. 
    #this is the most crucial [art of wolff algorithm, as it is the only cluster which is evaluated at one mc step.]
    for i in range(len(cluster_island)):
        if len(cluster_island[i]) == max([len(cluster) for cluster in cluster_island]):
            cluster = cluster_island[i]
            flipping_the_cluster = cluster_is_flipped[i]
            break
    cluster_island = [cluster]

    return cluster_island, spins, cluster_is_flipped


#############################################################################################################

#The basic perspective of both the wolff and sw algorithms are same, just the last function where, one need to evaluate the cluster, 
#there in sw one evaluate the group of clusters, whereas in wolff, there is only one cluster evaluated at one mc steps. 


def bonds_evalulation(params, spins):
    
    """
    This fucntion goes over all the spins in the system and checks, if there is opposite spins anywhere in the system, 
    and if opposite then it set bonds as 0, or if there are same spins, then it sets the bond as infinity with probability of 
    (1 - e^-2J/(k_b * T)).
    
    Args:
    spins : 2D array (L, L)
        contains all the spins in the system.
        
    It, returns the bonds in the system, which is a 3D array conataining (2, L, L) dimension.
    Firsy 2D array gives the horizontal bonds, and the second 2D array gives the vertical boind, 
    ofcourse, the bond is set to 0, if the spins are opposite, and bond is set to infinity, if the spins are same,
    with a Periodic boundary condition. 
    """
    

    bonds = np.zeros((2, params.L, params.L,),dtype=float)
    
    prob = np.minimum(1, np.exp(-2*params.J/(params.kb*params.T)))
    
    
    diff_horizontal_spin = np.abs(spins + np.roll(spins,-1,axis=1))/2            # we roll the spins, so that, if there is a same spin it will 
                                                                                 #become add to 2 or else, it will become 0
    spin_horizontal_indi = np.asarray(np.nonzero(diff_horizontal_spin))           # It returns 2 array which contains the labelling of matrix elements position 
                                                                               #where the spins are different

    diff_vertical_spin = np.abs(spins + np.roll(spins,-1,axis=0))/2            # we roll the spins, so that, if there is a same spin it will 
                                                                                 #become add to 2 or else, it will become 0
    spin_vertical_indi = np.asarray(np.nonzero(diff_vertical_spin))           # It returns 2 array which contains the labelling of matrix elements position 
                                                                               #where the spins are different

    for i in range(np.shape(spin_horizontal_indi)[1]):
        if np.random.binomial(1, prob) == 1:
            bonds[0, spin_horizontal_indi[0,i], spin_horizontal_indi[1,i]] = 0
        else:
            bonds[0, spin_horizontal_indi[0,i], spin_horizontal_indi[1,i]] = np.inf

    for j in range(np.shape(spin_vertical_indi)[1]):
        if np.random.binomial(1, prob) == 1:
            bonds[1, spin_vertical_indi[0,j], spin_vertical_indi[1,j]] = 0
        else:
            bonds[1, spin_vertical_indi[0,j], spin_vertical_indi[1,j]] = np.inf
    
    return bonds


#############################################################################################################



def back_tracking(params, x, y, bonds, n_visited, cluster, spins, flipping_the_cluster):
    
    """
    It takes the argument of x and y coordinates, 
    cluster, not visted sites, spins, and flipping the cluster which is a array containing 
    entries as 1 or --1, flipping the cluster with probability of 1/2. 
    """
    
    """
    The function takes the tracking of the cluster, and flips the spins in the cluster accordingly. 
    It search for the nearest neighbours of the spin, if they are found to be equal this functions 
    takes over to that spin and reiterate itself. Once visited spins are not visited second time, it keeps the track of this. 
    Whenever an bond is found, the corresponding spins are added to that cluster.
    """ 
    
    """

    Returns:
        cluster and spins after the flipping of the cluster,, or maybe not, because it is done with probability half,
        depending upon the entries in the array flipping_the_cluster
    """
    
    
    if n_visited[x, y]:
        n_visited[x, y] = False
        cluster.append([x, y])
        spins[x, y] = spins[x, y] * flipping_the_cluster
                
        if bonds[0][x][y] == np.inf:
            n_x = x
            n_y = (y + 1)%params.L
            cluster, spins = back_tracking(params, n_x, n_y, bonds, n_visited, cluster, spins, flipping_the_cluster)
            
        if bonds[0][x][(y - 1)%params.L] == np.inf:
            n_x = x
            n_y = (y - 1)%params.L
            cluster, spins = back_tracking(params, n_x, n_y, bonds, n_visited, cluster,  spins, flipping_the_cluster)
            
        if bonds[1][x][y] == np.inf:
            n_x = (x + 1)%params.L
            n_y = y
            cluster, spins = back_tracking(params, n_x, n_y, bonds, n_visited, cluster,  spins, flipping_the_cluster)
            
        if bonds[1][(x - 1)%params.L][y] == np.inf:
            n_x = (x - 1)%params.L
            n_y = y
            cluster, spins = back_tracking(params, n_x, n_y, bonds, n_visited, cluster,  spins, flipping_the_cluster)
            
    return cluster, spins


#############################################################################################################


