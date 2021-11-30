#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:15:43 2020

@author: thosvarley

A library for general analysis of Markov-chains based on state-transition dynamics. 
Ideally will ultimately a useful package for information-theoretic analysis of toy systems. 

Will port to Cython at some point. 
"""
import numpy as np 
from scipy.stats import entropy
from scipy import linalg
from itertools import product 
from collections import Counter
from copy import deepcopy
import dit 
import networkx as nx
import matplotlib.pyplot as plt

def make_microscale(mat, split, mapping):
    """
    A function that takesn in a "macro-scale" TPM and splits a single element in two, and then 
    returns the causally equivalent "micro-scale" TPM. The actual Boolean functions of the TPM 
    don't actually matter. 
    
    Arguments:
    
        mat: 
            A square matrix where the number of rows and columns is of the form 2^N, where N is an integer.
        split:
            An integer giving the index of the node to split in two. Must be less than log2(mat.shape[0])
        mapping:
            A dictionary that tells how pairs of micro-states for the split node map to macro-states.
    
    Returns:
    
        micro
            A square matrix where the number of rows is 2^(N+1). 
    """
    
    #Start by defining essentiall variables:
    num_elements = int(np.log2(mat.shape[0])) #The number of elements at the macro-scale.
    macro_nodes = np.array([bin(i)[2:].zfill(num_elements) for i in range(2**num_elements)]) #All macro-state configurations
    micro_nodes = np.array([bin(i)[2:].zfill(1+num_elements) for i in range(2**(1+num_elements))]) #All micro-state configurations
    state_mapping = {i : set() for i in macro_nodes} #A dictionary that maps macro-configurations to compatable micro-configurations.
    
    #Dict giving which macro states (keys) map to what sets of possible micro states (given as a set)
    for i in macro_nodes:
        for j in micro_nodes:
            if (mapping[j[split:split+2]] == i[split]) and (j[:split] == i[:split]) and (i[split+1:] == j[split+2:]):
                state_mapping[i].add(j)
    
    #It's easier to work w/ dictionaries rather than trying to do everything as a numpy array. 
    macro_tpm_dict = {(macro_nodes[i],macro_nodes[j]) : mat[i][j] for i in range(len(macro_nodes)) for j in range(len(macro_nodes))}
    micro_tpm_dict = {(micro_nodes[i],micro_nodes[j]) : 0 for i in range(len(micro_nodes)) for j in range(len(macro_nodes))}
    
    #We look at the probability associated with every macro-scale transition and redistribute it, edge by edge. 
    for row in macro_nodes:
        for col in macro_nodes:

            exp_rows = state_mapping[row]
            exp_cols = state_mapping[col]

            for r in exp_rows:
                for c in exp_cols:
                    micro_tpm_dict[(r, c)] = macro_tpm_dict[(row, col)] / len(exp_cols)
    
    micro = np.zeros((2**(num_elements+1),
                      2**(num_elements+1)))
                 
    for key in micro_tpm_dict.keys():
        micro[int(key[0],2)][int(key[1],2)] = micro_tpm_dict[key]

    
    return micro

def tpm_temporal_MI(dist, M, base=2):
    """
    Given a TPM and some input distribution, calculates the Mutual Information 
    between the present state and the future state.
    
    The input distribution can be any probability distribution (usually Hmax or the equilibrium distribution of M)
    """
    assert M.shape[0] == M.shape[1], "M must be a square matrix"
    assert len(dist) == M.shape[0], "The shape of M must be len(dist)**2"
    
    #Calculating the normalized joint disribution
    M_norm = deepcopy(M)
    for i in range(M_norm.shape[0]):
        M_norm[i] = M_norm[i] * dist[i]
    
    M_norm = M_norm / np.sum(M_norm)
    H_joint = entropy(M_norm.flatten(), base=base)
    
    #Calculating the entropy of the input distribution
    H_dist = entropy(dist, base=base)
    
    #Calculation the entropy of the future distribution
    future = np.matmul(dist, M)
    H_future = entropy(future, base=base)
    
    return H_dist + H_future - H_joint


def tpm_union_entropy(dist, M):
    
    assert M.shape[0] == M.shape[1], "M must be a square matrix"
    assert len(dist) == M.shape[0], "The shape of M must be len(dist)**2"
    
    #Calculating the normalized joint disribution
    M_norm = deepcopy(M)
    M_norm = M_norm / np.sum(M_norm)
        
    future = np.matmul(dist, M)
    future = future / np.sum(future)
    
    prod = list(product(range(future.shape[0]), repeat=2))
    union_ents = [M_norm[prod[i]] * max(-np.log2(dist[prod[i][0]]),
                                        -np.log2(future[prod[i][1]])) 
                  for i in range(len(prod))]
    
    return sum([x for x in union_ents if np.isnan(x) == False])

def equilibrium_dist(M, combine_dist=False):
    """
    Calculates the equilibrium distribution of a given Markov chain TPM
    as the normalized eigenvector of the largest left-eigenvalue of the TPM.
    
    Since all TPM rows sum to 1, the largest eigenvalue should always be 1 and have real-part = 0.
    
    This assumes that there is only 1 valid steady-state for the Markov Chain (i.e. only one eigenvalue == 1)
    If you have a system whith more than 1 eigenvalue == 1, then the system is non-ergodic.
    We deal with this by normalizing all valid eigenvectors and then averaging them together.
    This assumes that all initial conditions are equiprobable and captures the distributions of states over all initial conditions. 
    """
    w, vl, vr = linalg.eig(M, left=True)
    
    if 1 not in np.real(w):
        where = np.argmax(np.real(w))
        pi = np.real(np.squeeze(vl[:,where] / np.sum(vl[:,where])))
        pi = np.divide(pi, np.sum(pi))
        
    else:
        where = np.array(np.isclose(np.real(w), 1.0).nonzero())[0]
        
        if where.shape[0] == 1:
            pi = np.real(np.squeeze(vl[:,where] / np.sum(vl[:,where])))
            pi = np.divide(pi, np.sum(pi))
        else:
            if combine_dist == True:
                valid_vectors = vl[:, where]
                for i in range(where.shape[0]):
                    valid_vectors[:,i] = valid_vectors[:,i] / np.sum(valid_vectors[:,i])
            
                pi = np.mean(valid_vectors, axis=1)
            else:
                valid_vectors = vl[:, where]
                for i in range(where.shape[0]):
                    valid_vectors[:,i] = valid_vectors[:,i] / np.sum(valid_vectors[:,i])
                
                pi = valid_vectors
                
    return pi

def dit_dist(dist, M):
    """
    Given an input distribution and a TPM, return the DIT Distribution Object required
    for PID analysis and other fun things you can do with DIT. 
    """
    num_elements = int(np.log2(M.shape[0]))
    states = ["".join(x) for x in product(["0","1"], repeat=num_elements)]
    perms = []
    
    alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M",
                "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                "a","b","c","d","e","f","g","h","i","j","k","l","m"]
    
    for i in range(M.shape[0]):
        if M.shape[0] <= 10:
            perms += [x + str(i) for x in states]
        elif M.shape[0] > 10:
            perms += [x + alphabet[i] for x in states]
    
    M_norm = deepcopy(M)
    for i in range(dist.shape[0]):
        M_norm[i] = M_norm[i]*dist[i]
    
    M_norm = M_norm.T.flatten() / np.sum(M_norm)
    
    outcomes = {perms[i] : M_norm[i] for i in range(M_norm.shape[0])}
    
    return dit.Distribution(outcomes)
   

def pid_dicts(my_pid):
    """
    Given a DIT.PID object, returns two dictionaries. 
    
    One gives you the ordering of every PID atom (in descending order: 
        lattice_order[0] is the total synergy, lattice_order[N] is the total redundancy.
    The other gives you the total reundancy and the PI term for a given atom, indexed with it's string:
        pid_results["{0:1}"] = (red, PI)
    """
    rows = [x.split(" |") for x in my_pid.to_string().split("\n|")[2:]]
    lattice_order = {}
    pid_results = {}
    for i in range(len(rows)):
        lattice_order[i] = [x for x in rows[i][0].split(" ") if x != ""][0]
        pid_results[lattice_order[i]] = (float(rows[i][1]), float(rows[i][2]))
    
    return lattice_order, pid_results

