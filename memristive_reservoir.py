#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import random
import scipy.integrate as spint
from sklearn.linear_model import Ridge
import sys

def incidence(Adjacency_sparse):
    N = Adjacency_sparse.shape[0]
    M = Adjacency_sparse.nnz
    
    Incid = np.zeros((N,M))
    for node in range(N):
        beg_edge_idx, end_edge_idx = Adjacency_sparse.indptr[node], \
                                     Adjacency_sparse.indptr[node+1]
        connected_nodes = Adjacency_sparse.indices[beg_edge_idx:end_edge_idx]
        Incid[node,beg_edge_idx:end_edge_idx] = 1
        Incid[connected_nodes, range(beg_edge_idx,end_edge_idx)] = -1
    
    return Incid[:N-1, :]

def vertex_projector(Adjacency_sparse):
    
    Incid = incidence(Adjacency_sparse)
    BBT_inv = linalg.inv(Incid.dot(Incid.T))

    return np.einsum('ji,jk,kl', Incid, BBT_inv, Incid)


def cycle_projector(Adjacency_sparse):
    M = Adjacency_sparse.nnz
    return np.eye(M) - vertex_projector(Adjacency_sparse)


def create_adj_ER_F(n, m):
    edges = []
    num_edges = 0
    
    while num_edges < m:
        new_edge = random.sample(range(n), 2)
        new_edge.sort()
        if new_edge not in edges:
            edges.append(new_edge)
            num_edges += 1
    edges.sort()
    adjacency = sparse.lil_matrix((n, n), dtype='int')
    for i,j in edges:
        adjacency[i, j] = 1
    return adjacency.tocsr()

def create_adj_ER_G(n, p):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                edges.append([i, j])
            
    adjacency = sparse.lil_matrix((n, n), dtype='int')
    for i,j in edges:
        adjacency[i, j] = 1
    return adjacency.tocsr()

#==========================================================
# 2D Cubic
#==========================================================

def create_adj_cubic_2d(lattice_shape, undirected=True, xbias=1, ybias=1 ):
    """
    Returns an adjacency matrix for a 2D cubic lattice with number
    of nodes specified by lattice_shape.  If a directed network is
    requested with no bias, the default configuration is
    all bonds going from left to right and top to bottom. (recalling
    that we index nodes across rows then columns).  The xbias and
    ybias give the probability that a bond goes from left to
    right versus RL and top to bottom versus BT respectively.
    """
    num_ynodes, num_xnodes = lattice_shape
    num_nodes = num_xnodes * num_ynodes
    
    A = sparse.lil_matrix((num_nodes, num_nodes))
    
    # Form bond arrays to fill in row bonds and column bonds of the lattice
    x_bonds = np.ones(num_xnodes-1)
    y_bonds = np.ones(num_ynodes-1)
    
    # connect each row node to its neighbor to the right
    for first_row_node in range(0, num_nodes, num_xnodes):
         A[range(first_row_node, first_row_node + num_xnodes - 1),\
          range(first_row_node + 1, first_row_node + num_xnodes)] = x_bonds
    
    # connect each column node to its neighbor below
    for first_col_node in range(0, num_xnodes):
         A[range(first_col_node, num_nodes - num_xnodes, num_xnodes),\
          range(first_col_node + num_xnodes, num_nodes, num_xnodes)] = y_bonds
    
    # If we want an undirected network, just return the symmetrized form
    if undirected:
        A = A.tocsr()
        return A + A.T
    else:
        # If we want to toggle the direction of the elements (default direction is right and down)
        if (xbias != 1) or (ybias != 1):
            rows, cols = A.nonzero()
        
            for i, j in zip(rows, cols):
                if np.abs(i-j) == 1: # row bond
                    if np.random.rand() > xbias: # overcome the bias with probability 1-xbias
                        A[i, j] = 0
                        A[j, i] = 1
                else: #column bond
                    if np.random.rand() > ybias:
                        A[i, j] = 0
                        A[j, i] = 1
        return A.tocsr()

#==========================================================
# 2D Cubic Random
#==========================================================

def cubic_2d_random(lattice_shape, concentration, undirected=True, single_bond=False, xbias=1, ybias=1):
    """
    Returns a random 2d lattice with specified concentration in CSR format.  Besides an undirected
    network, we may also generate random directed networks of a specified concentration. The
    single_bond variable specified whether we may have bonds in both directions or only in one
    at a time. The xbias and ybias give the probability that a bond goes from left to
    right versus RL and top to bottom versus BT respectively.
    """
    # for an undirected network, we begin with a directed network, choose which bonds to keep and then symmetrize
    # Changing the sparsity structure of LIL matrices is faster
    if undirected:
        A = create_adj_cubic_2d(lattice_shape, undirected=False).tolil()
    # if we want a multiple bond network, we begin with a full undirected network
    elif not single_bond:
        A = create_adj_cubic_2d(lattice_shape).tolil()
    # for a single bond network, we begin with the directed network and then prune
    elif single_bond:
        A = create_adj_cubic_2d(lattice_shape, undirected=False, xbias=xbias, ybias=ybias).tolil()
    else:
        print("Invalid parameters defining lattice.  Check undirected and single_bond")
    
    # Get nonzero indices
    rows, cols = A.nonzero()
    # Loop over nonzero elements
    for i, j in zip(rows, cols):
        if np.random.rand() > concentration:   # Delete the bond with probability 1-concentration
            A[i, j] = 0
    
    A = A.tocsr()
    if undirected: # symmetrize before returning
        return A + A.T
    else:
        return A

#==========================================================
# Locally Connected
#==========================================================

def locally_connected(lattice_shape, distribution):
    """
    Returns an adjacency matrix for a lattice of size lattice_shape whose connectivity is specified by a local distribution.
    Distances between nodes are normalized such that the lattice spacing is 1. The distribution function gives the probability of
    a connection at a given distance.  The networks returned are undirected.
    """
    def node2xy(node_idx):
        """
        returns the x and y coordinates of a node index in our grid supposing that the 0,0 point is in the upper left
        and the positive y-axis points down
        """
        return node_idx % lattice_shape[1], int(node_idx / lattice_shape[1])
    
    def distance(nodei_idx, nodej_idx):
        """
        Returns the distance between nodes i and j assuming a cubic lattice indexed across rows
        """
        x1, y1 = node2xy(nodei_idx)
        x2, y2 = node2xy(nodej_idx)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    num_nodes = lattice_shape[0] * lattice_shape[1]
    
    A = sparse.lil_matrix((num_nodes, num_nodes), dtype='float')
    
    for node_i in range(num_nodes):
        for node_j in range(node_i+1, num_nodes):
            if np.random.rand() < distribution(distance(node_i, node_j)):
                A[node_i, node_j] = 1
    
    return (A + A.T).tocsr()


def CaravelliEqn(source_func, Omega_A, alpha, beta, chi):
    M = Omega_A.shape[0]
    def dwdt(t, w, above_mask, below_mask):
        Curr_proj = linalg.inv(np.eye(M) - chi*np.dot(Omega_A,np.diag(w)))
        deriv = -alpha*w - 1/beta*np.einsum('jk,k', Curr_proj, source_func(t))
        above_halt = np.logical_and(above_mask, deriv > 0)
        below_halt = np.logical_and(below_mask, deriv < 0)
        deriv[above_halt] = 0
        deriv[below_halt] = 0
        return deriv
    return dwdt


def CaravelliEqn_windowed_biolek(source_func, Omega_A, alpha, beta, chi):
    M = Omega_A.shape[0]
    def dwdt(t, w):
        Curr_proj = linalg.inv(np.eye(M) - chi*np.dot(Omega_A,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, source_func(t))
        deriv = -alpha*w + I/beta
        deriv = deriv * (1-(w-np.heaviside(-I, 0))**4)
        return deriv
    return dwdt


def CaravelliEqn_windowed_joglekar(source_func, Omega_A, alpha, beta, chi):
    M = Omega_A.shape[0]
    def dwdt(t, w):
        Curr_proj = linalg.inv(np.eye(M) - chi*np.dot(Omega_A,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, source_func(t))
        deriv = -alpha*w + I/beta
        deriv = deriv * (1-(2*w-1)**4)
        return deriv
    return dwdt

def integrate_Caravelli(dwdt, t_interval, w0):
    m = w0.shape[0]
    def w_above(t, w, above_mask):
        return np.max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return np.min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros_like(w0)
    below_mask = np.zeros_like(w0)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = w0.copy().reshape(m, 1)
    epsilon = 1e-8
    while t < t_final:
        dwdt_masked = lambda t, w: dwdt(t, w, above_mask, below_mask)
        w_above_masked = lambda t, w: w_above(t, w, above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, w: w_below(t, w, below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dwdt_masked, (t, t_final), w0, events=[w_above_masked, w_below_masked], max_step = 0.05)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        w0 = sol.y[:,-1].copy()
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj

def integrate_Caravelli_windowed(dwdt, t_interval, w0):

    t, t_final = t_interval

    sol = spint.solve_ivp(dwdt, (t, t_final), w0, max_step = 0.05)
       
    return sol.t, sol.y

def RCEqn(source, B, R, C):
    '''
    R and C are vectors
    '''
    BCBt = linalg.inv(np.einsum('ij,jk,lk', B, np.diag(C), B))
    BtBCBtBR = np.einsum('ji,jk,kl,lm', B, BCBt, B, np.diag(1/R))
    def dvdt(t, v):
        
        deriv = np.dot(BtBCBtBR, (source(t) - v))
        return deriv
    return dvdt

def integrate_RC(dvdt, t_interval, v0):
    m = v0.shape[0]
    t, t_final = t_interval
    
    times = np.array([0])
    sol = spint.solve_ivp(dvdt, t_interval, v0, max_step=0.05)
    
    return sol.t, sol.y




def meansquare_function(time_series, times):
    T = times[-1] - times[0]
    dt = times[1:] - times[:-1]
    squared = time_series**2
    squared_symm = (squared[:-1] + squared[1:])/2
    return np.dot(squared_symm, dt)/T

def MSE_reservoir(training_data, reconstruction, times):
    return meansquare_function(training_data - reconstruction, times)

def train_reservoir(times, traj, output_func, transient, training_tfinal, ridge_param=0.0001):
    '''
    Return the trained regressor and reconstruction of the output function
    '''
    
    training_mask = np.logical_and(times > transient, times <= training_tfinal)
    training_times = times[training_mask]
    training_traj = traj[:, training_mask]
    num_times = np.shape(training_traj)[1]
    training_traj = np.append(training_traj, np.ones((1, num_times)), axis=0)
                                    
    training_output = output_func(training_times)
    # Necessary to have MSE bounded
    normalization = np.sqrt(meansquare_function(training_output, training_times))
    training_output = training_output/normalization
    
    clf = Ridge(alpha=ridge_param)
    clf.fit(training_traj.T, training_output)
    reconstruction = clf.predict(training_traj.T)
    MSE_train = MSE_reservoir(training_output, reconstruction, training_times)
    
    return clf, training_times, reconstruction, MSE_train
##
##def train_reservoir_fixed(times, traj, output_func, transient, training_tfinal, ridge_param=0.0001):
##    '''
##    Return the trained regressor and reconstruction of the output function
##    '''
##    
##    training_mask = np.logical_and(times > transient, times <= training_tfinal)
##    training_times = times[training_mask]
##    training_traj = traj[:, training_mask]
##    num_times = np.shape(training_traj)[1]
##    training_traj = np.append(training_traj, np.ones((1, num_times)), axis=0)
##                                    
##    training_output = output_func(training_times)
##    # Necessary to have MSE bounded
##    z_square = meansquare_function(training_output, training_times)
##    training_output = training_output
##    
##    clf = Ridge(alpha=ridge_param)
##    clf.fit(training_traj.T, training_output)
##    reconstruction = clf.predict(training_traj.T)
##    MSE_train = MSE_reservoir(training_output, reconstruction, training_times) / z_square
##    
##    return clf, training_times, reconstruction, MSE_train


def reconstruct(times, traj, clf, tbegin, tfinal):
    recon_mask = np.logical_and(times > tbegin, times < tfinal)
    recon_times = times[recon_mask]
    recon_traj = traj[:, recon_mask]
    recon_traj = np.append(recon_traj, np.ones((1, len(recon_times))), axis=0)
    return recon_times, clf.predict(recon_traj.T)

def LRCEqn(source_vec_func, B, L, R, C, R_C):
    '''
    L, R, C and R_C are vectors
    '''
    BRinvBt_inv = linalg.inv(np.einsum('ij,j,kj->ik', B, 1/R, B))
    Omega_BRinv = np.einsum('ji,jk,kl,l->il', B, BRinvBt_inv, B, 1/R)
    LCinv = 1/(L*C)
    LinvOmega = np.einsum('i, ij->ij', 1/L, Omega_BRinv)
    LinvOmegaRRC = np.einsum('j,jk->jk', 1/L, np.dot(Omega_BRinv, np.diag(R)) + np.diag(R_C))
    def dLRCdt(t, ab_vec):
        M = int(ab_vec.size/2)
        a = ab_vec[:M]
        b = ab_vec[M:]
        adot = b
        bdot = -np.einsum('i,i->i', LCinv, a) - np.dot(LinvOmegaRRC, b) - np.dot(LinvOmega, source_vec_func(t))
        return np.concatenate((adot, bdot))
    return dLRCdt

def integrate_LRC(dLRCdt, t_interval, ab_0):

    sol = spint.solve_ivp(dLRCdt, t_interval, ab_0, max_step=0.05)
    
    return sol.t, sol.y

def LRC_spectrum(B, L, R, C, R_C):
    M = L.size
    BRinvBt_inv = linalg.inv(np.einsum('ij,j,kj->ik', B, 1/R, B))
    Omega_BRinv = np.einsum('ji,jk,kl,l->il', B, BRinvBt_inv, B, 1/R)
    
    LCinv = 1/(L*C)
    LinvOmegaRRC = np.einsum('j,jk->jk', 1/L, np.dot(Omega_BRinv, np.diag(R)) + np.diag(R_C))
    return linalg.eigvals(np.block([[np.zeros((M, M)), np.eye(M)],
                                    [-np.diag(LCinv), -LinvOmegaRRC]
                                    ]))

def memory_function(times, traj, output_func, delays, transient, training_tfinal):
    mem_func_train = np.zeros_like(delays, dtype='float')

    for idx, tau in enumerate(delays):
        
        def delayed_output(t):
            return output_func(t-tau)
        
        clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, delayed_output, transient+tau, training_tfinal)
    
        mem_func_train[idx] = 1 - MSE_train
       
    return mem_func_train



def find_capacity(times, traj, output_func, threshold, transient, training_tfinal, tol = 1):
    '''
    The capacity is the smallest delay for which the reconstruction error is greater than
    the threshold
    '''
    max_capacity = 1e8
    min_capacity = 0
    delay = 0.3
    while (max_capacity - min_capacity) > tol:
        
        if training_tfinal - (transient+delay) < 200:
            sys.exit('Training period insufficient')
            
        def delayed_output(t):
            return output_func(t-delay)
        
        clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, delayed_output, transient+delay, training_tfinal)

        if MSE_train < threshold:
            min_capacity = delay
            if 2*delay < max_capacity:
                delay = 2*delay
            else:
                delay = (max_capacity + delay)/2
        else:
            max_capacity = delay
            delay = (delay + min_capacity)/2
    return (min_capacity + max_capacity)/2


def find_capacity_2(times, traj, output_func, threshold, transient, training_tfinal):
    
    # first find the delay for which the diagonal reconstruction has greater error than
    # threshold
    def square_func(t):
        return output_func(t) * output_func(t)
    diag_cap = find_capacity(times, traj, square_func, threshold, transient, training_tfinal, tol=0.02)
    
    # cover (2*diag_cap, 2*diag_cap) square with squares and count number over threshold
    approx_step_size = 0.5
    num_steps = int(2*diag_cap/approx_step_size)

    
    delays = np.linspace(0, 2*diag_cap, num=num_steps)
    step_size = delays[1] - delays[0]
    area = step_size**2
    
    square_capacity = np.zeros((len(delays), len(delays)))
    
    for idx1, tau1 in enumerate(delays):
        for idx2, tau2 in enumerate(delays[idx1:]):
        
            def quad_delayed_noise(t):
                return output_func(t-tau1) * output_func(t-tau2)
        
            transient = 100 + max(tau1, tau2)
            clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quad_delayed_noise, transient, training_tfinal)

            square_capacity[idx1, idx1+idx2] = 1 - MSE_train
        
        
    square_capacity = square_capacity + square_capacity.T

    np.fill_diagonal(square_capacity, square_capacity.diagonal()/2)
    
    return np.count_nonzero(square_capacity > 1-threshold)*area



def gen_adj_triangular(lattice_shape, undirected=True, xbias=1, ybias=1):
    num_ynodes, num_xnodes = lattice_shape
    num_nodes = num_xnodes * num_ynodes
    
    A = sparse.lil_matrix((num_nodes, num_nodes))
    
    x_bonds = np.ones(num_xnodes-1)
    y_bonds = np.ones(num_ynodes-1)
    
    #print "Rows"
    for first_row_node in range(0, num_nodes, num_xnodes):
        A[range(first_row_node, first_row_node + num_xnodes - 1),\
          range(first_row_node + 1, first_row_node + num_xnodes)] = x_bonds
        #print first_row_node, range(first_row_node, first_row_node + num_xnodes - 1),\
         # range(first_row_node + 1, first_row_node + num_xnodes)
    
    # connect each column node to its neighbor below
    #print "Cols"
    for first_col_node in range(0, num_xnodes):
        A[range(first_col_node, num_nodes - num_xnodes, num_xnodes),\
          range(first_col_node + num_xnodes, num_nodes, num_xnodes)] = y_bonds
        #print first_col_node, range(first_col_node, num_nodes - num_xnodes, num_xnodes),\
        #  range(first_col_node + num_xnodes, num_nodes, num_xnodes)
    
    # connect every even row (except first node) to it's neighbor 1 back and below
    #print "Even Diag"
    for first_even_row_node in range(0, num_nodes-num_xnodes, 2*num_xnodes):
        A[range(first_even_row_node+1, first_even_row_node + num_xnodes),\
          range(first_even_row_node + num_xnodes, first_even_row_node + 2*num_xnodes - 1)] = x_bonds
        #print first_even_row_node, range(first_even_row_node+1, first_even_row_node + num_xnodes),\
        #  range(first_even_row_node + num_xnodes, first_even_row_node + 2*num_xnodes - 1)
        
    # connect every odd row (except last node) to its neighbor 1 forward and below
    #print "Odd Diag"
    if num_ynodes > 2:
        for first_odd_row_node in range(num_xnodes, num_nodes-num_xnodes, 2*num_xnodes):
            A[range(first_odd_row_node, first_odd_row_node + num_xnodes - 1),\
              range(first_odd_row_node + num_xnodes + 1, first_odd_row_node + 2*num_xnodes)] = x_bonds
            #print first_odd_row_node, range(first_odd_row_node, first_odd_row_node + num_xnodes - 1),\
            #  range(first_odd_row_node + num_xnodes + 1, first_odd_row_node + 2*num_xnodes)
    
    # If we want an undirected network, just return the symmetrized form
    if undirected:
        A = A.tocsr()
        return A + A.T
    else:
        # If we want to toggle the direction of the elements (default direction is right and down)
        if (xbias != 1) or (ybias != 1):
            rows, cols = A.nonzero()
        
            for i, j in zip(rows, cols):
                if np.abs(i-j) == 1: # row bond
                    if np.random.rand() > xbias: # overcome the bias with probability 1-xbias
                        A[i, j] = 0
                        A[j, i] = 1
                else: #column bond
                    if np.random.rand() > ybias:
                        A[i, j] = 0
                        A[j, i] = 1
        return A.tocsr()

def Diluted_MEM_LRCEqn(source_vec_func, B, L, C, R_C, R_func, R_off, MemR_mask, alpha, beta):
    '''
    L, R, C and R_C are vectors.  MemR is a binary vector 
    '''
    
    
    BBt_inv = linalg.inv(np.einsum('ij,kj', B, B))
    Omega_B = np.einsum('ji,jk,kl', B, BBt_inv, B)
    LCinv = 1/(L*C)
    def dMEM_LRCdt(t, abw_vec, above_mask, below_mask):
        num_mem = len(above_mask)
        M = int((len(abw_vec) - num_mem)/2)
        a = abw_vec[:M]
        b = abw_vec[M:2*M]
        w = abw_vec[2*M:]
        
        R_mem = R_func(w)
        R = R_off * np.ones(M)
        R[MemR_mask] = R_mem
        
        Rinv = 1/R
        Linv = 1/L
        Omega_BRinv_R = linalg.inv(np.eye(M) + np.einsum('ij,j->ij', Omega_B, 1/R-1)).dot(Omega_B)
        Omega_BRinv = np.einsum('ij,j->ij', Omega_BRinv_R, 1/R)
        
        adot = b
        
        bdot = -np.einsum('i,i->i', LCinv, a)
        bdot += -np.einsum('i,ij, j->i', Linv, Omega_BRinv_R + np.diag(R_C), b)
        bdot += -np.einsum('i,ij,j->i', Linv, Omega_BRinv, source_vec_func(t))
        
        wdot = -R_off/beta * np.einsum('i,ij,j->i', Rinv, Omega_BRinv, b)[MemR_mask]
        wdot += -alpha * w
        wdot += R_off/beta * np.einsum('i,ij,j->i', Rinv, np.eye(M) - Omega_BRinv, source_vec_func(t))[MemR_mask]
        
        above_halt = np.logical_and(above_mask, wdot > 0)
        below_halt = np.logical_and(below_mask, wdot < 0)
        wdot[above_halt] = 0
        wdot[below_halt] = 0
        return np.concatenate((adot, bdot, wdot))
    return dMEM_LRCdt

def integrate_dilute_MEMLRC(dMLRCdt, t_interval, abw0, m):
    
    def empty_min(x):
        # If all memristors are in bottom state keep integrating
        if x.size == 0:
            return 0.5
        else:
            return np.min(x)

    def empty_max(x):
        if x.size == 0:
            return 0.5
        else:
            return np.max(x)
        
    num_mem = len(abw0) - 2*m
    def w_above(t, w, above_mask):
        return empty_max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return empty_min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros(num_mem)
    below_mask = np.zeros(num_mem)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = abw0.copy().reshape(2*m+num_mem, 1)
    epsilon = 1e-8
    while t < t_final:
        dMLRCdt_masked = lambda t, abw: dMLRCdt(t, abw, above_mask, below_mask)
        w_above_masked = lambda t, abw: w_above(t, abw[2*m:], above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, abw: w_below(t, abw[2*m:], below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dMLRCdt_masked, (t, t_final), abw0, events=[w_above_masked, w_below_masked], max_step = 0.05)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        abw0 = sol.y[:,-1].copy()
        w0 = abw0[2*m:]
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj


def Diluted_MEM_LRCEqn_windowed(source_vec_func, B, L, C, R_C, R_func, R_off, MemR_mask, alpha, beta):
    '''
    L, R, C and R_C are vectors.  MemR is a binary vector 
    '''
    
    def window(w, I, wdot):
        window = (1-(w-np.heaviside(-I, 0))**4)
        window[w>=1] = -wdot[w>=1]
        window[w<=0] = wdot[w<=0]
        return window
    BBt_inv = linalg.inv(np.einsum('ij,kj', B, B))
    Omega_B = np.einsum('ji,jk,kl', B, BBt_inv, B)
    LCinv = 1/(L*C)
    def dMEM_LRCdt(t, abw_vec):
        M = len(LCinv)
        num_mem = len(abw_vec) - 2*M
        a = abw_vec[:M]
        b = abw_vec[M:2*M]
        w = abw_vec[2*M:]
        
        R_mem = R_func(w)
        R = R_off * np.ones(M)
        R[MemR_mask] = R_mem
        
        Rinv = 1/R
        Linv = 1/L
        Omega_BRinv_R = linalg.inv(np.eye(M) + np.einsum('ij,j->ij', Omega_B, 1/R-1)).dot(Omega_B)
        Omega_BRinv = np.einsum('ij,j->ij', Omega_BRinv_R, 1/R)
        
        adot = b
        
        bdot = -np.einsum('i,i->i', LCinv, a)
        bdot += -np.einsum('i,ij, j->i', Linv, Omega_BRinv_R + np.diag(R_C), b)
        bdot += -np.einsum('i,ij,j->i', Linv, Omega_BRinv, source_vec_func(t))

        I = -np.einsum('i,ij,j->i', Rinv, Omega_BRinv, b)[MemR_mask]
        I += np.einsum('i,ij,j->i', Rinv, np.eye(M) - Omega_BRinv, source_vec_func(t))[MemR_mask]
        wdot = R_off/beta * I
        wdot += -alpha * w

        wfunc = window(w, I, wdot) 
        wdot = wdot * wfunc
        return np.concatenate((adot, bdot, wdot))
    return dMEM_LRCdt

def integrate_dilute_MEMLRC_windowed(dMLRCdt, t_interval, abw0):
    
    t, t_final = t_interval
    
    sol = spint.solve_ivp(dMLRCdt, (t, t_final), abw0, max_step = 0.05)

        
    return sol.t, sol.y




def Mem1_LRC2_network(source_vec_func, MemNet1, LRCNet2, connections):
    Omega_A1, alpha, beta, chi, Roff = MemNet1
    B, L, R, C, R_C = LRCNet2
    M1 = Omega_A1.shape[0]
    M2 = B.shape[1]
    def dLayer1dt(t, w):
        Curr_proj = 1/Roff*linalg.inv(np.eye(M1) - chi*np.dot(Omega_A1,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, source_vec_func(t))
        deriv = -alpha*w + Roff/beta*I
        V = I / (Roff*(1-chi*w))
        return deriv, V

    BRinvBt_inv = linalg.inv(np.einsum('ij,j,kj->ik', B, 1/R, B))
    Omega_BRinv = np.einsum('ji,jk,kl,l->il', B, BRinvBt_inv, B, 1/R)
    LCinv = 1/(L*C)
    LinvOmega = np.einsum('i, ij->ij', 1/L, Omega_BRinv)
    LinvOmegaRRC = np.einsum('j,jk->jk', 1/L, np.dot(Omega_BRinv, np.diag(R)) + np.diag(R_C))
    def dLayer2dt(t, ab_vec, Layer1source):
        M = int(ab_vec.size/2)
        a = ab_vec[:M2]
        b = ab_vec[M2:]
        adot = b
        bdot = -np.einsum('i,i->i', LCinv, a) - np.dot(LinvOmegaRRC, b) - np.dot(LinvOmega, Layer1source)
        return np.concatenate((adot, bdot))

    def dNetworkdt(t, wab_vec, above_mask, below_mask):
        w = wab_vec[:M1]
        ab = wab_vec[M1:]
        deriv1, V1 = dLayer1dt(t, w)
        Layer1source = np.dot(connections, V1)
        deriv2 = dLayer2dt(t, ab, Layer1source)

        above_halt = np.logical_and(above_mask, deriv1 > 0)
        below_halt = np.logical_and(below_mask, deriv1 < 0)
        deriv1[above_halt] = 0
        deriv1[below_halt] = 0
        return np.concatenate((deriv1, deriv2))
    return dNetworkdt

def Integrate_MemR1_LRC2_net(dNetdt, t_interval, wab0, m1):
    
    
    def empty_min(x):
        # If all memristors are in bottom state keep integrating
        if x.size == 0:
            return 0.5
        else:
            return np.min(x)

    def empty_max(x):
        if x.size == 0:
            return 0.5
        else:
            return np.max(x)
        
    def w_above(t, w, above_mask):
        return empty_max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return empty_min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros(m1)
    below_mask = np.zeros(m1)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = wab0.copy().reshape(len(wab0), 1)
    epsilon = 1e-6
    while t < t_final:
        dNetdt_masked = lambda t, wab: dNetdt(t, wab, above_mask, below_mask)
        w_above_masked = lambda t, wab: w_above(t, wab[:m1], above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, wab: w_below(t, wab[:m1], below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dNetdt_masked, (t, t_final), wab0, events=[w_above_masked, w_below_masked], max_step = 0.1)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        wab0 = sol.y[:,-1].copy()
        w0 = wab0[:m1]
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj


def LRC1_Mem2_network(source_vec_func, LRCNet1, MemNet2, connections):
    Omega_A2, alpha, beta, chi, Roff = MemNet2
    B, L, R, C, R_C = LRCNet1
    M2 = Omega_A2.shape[0]
    M1 = B.shape[1]

    BRinvBt_inv = linalg.inv(np.einsum('ij,j,kj->ik', B, 1/R, B))
    Omega_BRinv = np.einsum('ji,jk,kl,l->il', B, BRinvBt_inv, B, 1/R)
    LCinv = 1/(L*C)
    LinvOmega = np.einsum('i, ij->ij', 1/L, Omega_BRinv)
    LinvOmegaRRC = np.einsum('j,jk->jk', 1/L, np.dot(Omega_BRinv, np.diag(R)) + np.diag(R_C))
    def dLayer1dt(t, ab_vec):
        a = ab_vec[:M1]
        b = ab_vec[M1:]
        adot = b
        bdot = -np.einsum('i,i->i', LCinv, a) - np.dot(LinvOmegaRRC, b) - np.dot(LinvOmega, source_vec_func(t))
        return np.concatenate((adot, bdot)), R_C*b

    def dLayer2dt(t, w, Layer1source):
        Curr_proj = 1/Roff*linalg.inv(np.eye(M1) - chi*np.dot(Omega_A2,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, Layer1source)
        deriv = -alpha*w + Roff/beta*I
        return deriv


    def dNetworkdt(t, abw_vec, above_mask, below_mask):
        w = abw_vec[2*M1:]
        ab = abw_vec[:2*M1]
        
        deriv1, VR_C = dLayer1dt(t, ab)
        Layer1source = np.dot(connections, VR_C)
        deriv2 = dLayer2dt(t, w, Layer1source)

        above_halt = np.logical_and(above_mask, deriv2 > 0)
        below_halt = np.logical_and(below_mask, deriv2 < 0)
        deriv2[above_halt] = 0
        deriv2[below_halt] = 0
        return np.concatenate((deriv1, deriv2))
    return dNetworkdt

def Integrate_LRC1_MemR2_net(dNetdt, t_interval, abw0, m2):
    
    
    def empty_min(x):
        # If all memristors are in bottom state keep integrating
        if x.size == 0:
            return 0.5
        else:
            return np.min(x)

    def empty_max(x):
        if x.size == 0:
            return 0.5
        else:
            return np.max(x)
        
    def w_above(t, w, above_mask):
        return empty_max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return empty_min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros(m2)
    below_mask = np.zeros(m2)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = abw0.copy().reshape(len(abw0), 1)
    epsilon = 1e-6
    while t < t_final:
        dNetdt_masked = lambda t, abw: dNetdt(t, abw, above_mask, below_mask)
        w_above_masked = lambda t, abw: w_above(t, abw[-m2:], above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, abw: w_below(t, abw[-m2:], below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dNetdt_masked, (t, t_final), abw0, events=[w_above_masked, w_below_masked], max_step = 0.1)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        abw0 = sol.y[:,-1].copy()
        w0 = abw0[-m2:]
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj

def Memnet_2Layer(source_vec_func, MemNet1, MemNet2, chi, Roff, connections):
    Omega_A1, alpha1, beta1 = MemNet1
    Omega_A2, alpha2, beta2 = MemNet1

    M1 = Omega_A1.shape[0]
    M2 = Omega_A2.shape[0]
    
    def dLayer1dt(t, w):
        Curr_proj = 1/Roff*linalg.inv(np.eye(M1) - chi*np.dot(Omega_A1,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, source_vec_func(t))
        deriv = -alpha1*w + Roff/beta1*I
        return deriv, I

    def dLayer2dt(t, w, Layer1source):
        Curr_proj = 1/Roff*linalg.inv(np.eye(M2) - chi*np.dot(Omega_A2,np.diag(w)))
        I = -np.einsum('jk,k', Curr_proj, Layer1source)
        deriv = -alpha2*w + Roff/beta2*I
        return deriv

    def dNetworkdt(t, w_vec, above_mask, below_mask):
        w1 = w_vec[:M1]
        w2 = w_vec[M1:]
        deriv1, I1 = dLayer1dt(t, w1)
        Layer1source = np.dot(connections, I1)
        deriv2 = dLayer2dt(t, w2, Layer1source)
        deriv = np.concatenate((deriv1, deriv2))

        above_halt = np.logical_and(above_mask, deriv > 0)
        below_halt = np.logical_and(below_mask, deriv < 0)
        deriv[above_halt] = 0
        deriv[below_halt] = 0
        return deriv
    return dNetworkdt
   
def Integrate_2Layer_Memnet(dNetdt, t_interval, w0, m1):
    
    def empty_min(x):
        # If all memristors are in bottom state keep integrating
        if x.size == 0:
            return 0.5
        else:
            return np.min(x)

    def empty_max(x):
        if x.size == 0:
            return 0.5
        else:
            return np.max(x)
        
    def w_above(t, w, above_mask):
        return empty_max(w[np.logical_not(above_mask)]) - 1
    w_above.terminal = True

    def w_below(t, w, below_mask):
        return empty_min(w[np.logical_not(below_mask)])
    w_below.terminal = True
    
    above_mask = np.zeros_like(w0)
    below_mask = np.zeros_like(w0)
    
    t, t_final = t_interval
    
    times = np.array([0])
    traj = w0.copy().reshape(len(w0), 1)
    epsilon = 1e-6
    while t < t_final:
        dNetdt_masked = lambda t, w: dNetdt(t, w, above_mask, below_mask)
        w_above_masked = lambda t, w: w_above(t, w, above_mask)
        w_above_masked.terminal = True
        w_below_masked = lambda t, w: w_below(t, w, below_mask)
        w_below_masked.terminal = True
        sol = spint.solve_ivp(dNetdt_masked, (t, t_final), w0, events=[w_above_masked, w_below_masked], max_step = 0.1)
        times = np.concatenate((times, sol.t[1:]))
        t = sol.t[-1]
        traj = np.hstack((traj, sol.y[:,1:]))
        w0 = sol.y[:,-1].copy()
        above_mask = (w0 >= 1-epsilon)
        w0[above_mask] = 1
        below_mask = (w0 <= epsilon)
        w0[below_mask] = 0
        
    return times, traj

def avg_capacity_2(times, traj, total_delay, noise_func):
    N = int(max((total_delay * 10)/2, 10))
    dt = total_delay / (2*N)
    
    capacity = 0
    num_capacities = 0

    
    for n1 in range(N+1):
        n2 = 2*N - n1
        
        def quad_delayed_noise(t):
            return noise_func(t-n1*dt) * noise_func(t-n2*dt)
        
        transient = 100 + n2*dt
        clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quad_delayed_noise, transient, times[-1])

        capacity += (1 - MSE_train)
        num_capacities += 1
    return capacity / num_capacities


def avg_capacity_3(times, traj, total_delay, noise_func):
    N = int(max((total_delay * 10)/3, 10))
    dt = total_delay / (3*N)
    
    capacity = 0
    num_capacities = 0
    
    for n1 in range(N+1):
        for n2 in range(n1, int((3*N - n1)/2)+1):
            n3 = 3*N - n1 - n2
            
            def cube_delayed_noise(t):
                return noise_func(t-n1*dt) * noise_func(t-n2*dt) * noise_func(t-n3*dt)
            
            transient = 100 + n3*dt
            clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, cube_delayed_noise, transient, times[-1])

            capacity += (1 - MSE_train)
            num_capacities += 1
            
    return capacity / num_capacities

def avg_capacity_4(times, traj, total_delay, noise_func):
    N = int(max((total_delay * 10)/4, 10))
    dt = total_delay / (4*N)
    
    capacity = 0
    num_capacities = 0
    
    for n1 in range(N+1):
        for n2 in range(n1, int((4*N - n1)/3)+1):
            for n3 in range(n2, int((4*N - n1 - n2)/2)+1):
                n4 = 4*N - n1 - n2 -n3
            
                def quart_delayed_noise(t):
                    return noise_func(t-n1*dt) * noise_func(t-n2*dt) * noise_func(t-n3*dt) * noise_func(t-n4*dt)
            
                transient = 100 + n4*dt
                clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quart_delayed_noise, transient, times[-1])

                capacity += (1 - MSE_train)
                num_capacities += 1
            
    return capacity / num_capacities

def diag_capacity_2(times, traj, total_delay, noise_func):
    tau = total_delay/2
        
    def quad_delayed_noise(t):
        return noise_func(t-tau) * noise_func(t-tau)
        
    transient = 100 + tau
    clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quad_delayed_noise, transient, times[-1])

    return (1 - MSE_train)

def diag_capacity_3(times, traj, total_delay, noise_func):
    tau = total_delay/3
        
    def cube_delayed_noise(t):
        return noise_func(t-tau) * noise_func(t-tau) * noise_func(t-tau)
        
    transient = 100 + tau
    clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, cube_delayed_noise, transient, times[-1])

    return (1 - MSE_train)

def diag_capacity_4(times, traj, total_delay, noise_func):
    tau = total_delay/4
        
    def quart_delayed_noise(t):
        return noise_func(t-tau) * noise_func(t-tau) * noise_func(t-tau) * noise_func(t-tau)
        
    transient = 100 + tau
    clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quart_delayed_noise, transient, times[-1])

    return (1 - MSE_train)

def quad_capacity(times, traj, output_func, delays, transient, training_tfinal):
    quad_capacity = np.zeros((len(delays), len(delays)))

    for idx1, tau1 in enumerate(delays):
        for idx2, tau2 in enumerate(delays[idx1:]):
        
            def quad_delayed_func(t):
                return output_func(t-tau1) * output_func(t-tau2)
        
            quad_transient = transient + max(tau1, tau2)
            clf, train_times, reconstruction, MSE_train = train_reservoir(times, traj, quad_delayed_func, quad_transient, training_tfinal)

            quad_capacity[idx1, idx1+idx2] = 1 - MSE_train
        
        
    quad_capacity = quad_capacity + quad_capacity.T
    np.fill_diagonal(quad_capacity, quad_capacity.diagonal()/2)
    return quad_capacity
