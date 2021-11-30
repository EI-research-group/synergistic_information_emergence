#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:09:42 2021

@author: tvarley
"""

import os
import numpy as np 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt
import seaborn as sns

lattice_dir = '/geode2/home/u100/tvarley/Carbonate/Libraries/PID_emergence/'
lattice = nx.read_gml("{0}lattice_3.gml".format(lattice_dir))
layers = nx.shortest_path_length(G=lattice, source="{0}{1}{2}")

pos = {
    "{0:1:2}" : (0,0),
    "{0:1}" : (-2,-2),
    "{0:2}" : (0,-2),
    "{1:2}" : (2,-2),
    "{0:1}{0:2}" : (-2,-4),
    "{0:1}{1:2}" : (-0,-4),
    "{0:2}{1:2}" : (2,-4),
    "{0}" : (-2,-6),
    "{1}" : (-0,-6),
    "{2}" : (2,-6),
    "{0:1}{0:2}{1:2}" : (4,-6),
    "{0}{1:2}" : (-2,-8),
    "{1}{0:2}" : (0,-8),
    "{2}{0:1}" : (2,-8),
    "{0}{1}" : (-2,-10),
    "{0}{2}" : (0,-10),
    "{1}{2}" : (2,-10),
    "{0}{1}{2}" : (0,-12),
    }

def synergy_bias(pid, lattice, layers):
    
    totals = {i : 0.0 for i in range(max(layers.values())+1)}
    mi = pid["{0:1:2}"][0]
    for i in range(max(layers.values())+1):
        totals[i] += sum({pid[x][1] for x in pid.keys() if layers[x] == i})/mi
    spectrum = np.array(list(totals.values()))
    
    dist = np.arange(spectrum.shape[0])
    bias = (dist*spectrum).sum() / dist.max()
    
    return bias, spectrum

gradient_dir = '/geode2/home/u100/tvarley/Carbonate/Libraries/PID_emergence/results/gradient/pid_3/'
listdir = os.listdir(gradient_dir)
gradient_syns = np.zeros(len(listdir))

tensor = np.load(gradient_dir + "/../tensor_3.npz")["arr_0"]

max_syn = 0.0
max_syn_pid = None
max_syn_file = None

for i in range(gradient_syns.shape[0]):
    pid = np.load("{0}{1}".format(gradient_dir, listdir[i]),
                  allow_pickle=True)["arr_0"].item()
    gradient_syns[i] = synergy_bias(pid, lattice, layers)[0]
    
    if gradient_syns[i] > max_syn:
        max_syn = gradient_syns[i]
        max_syn_pid = pid
        max_syn_file = listdir[i]

max_syn_spect = synergy_bias(max_syn_pid, lattice, layers)[1]
max_syn_tpm = tensor[:,:, 168]

min_syn = 1.0
min_syn_pid = None
min_syn_file = None

for i in range(gradient_syns.shape[0]):
    pid = np.load("{0}{1}".format(gradient_dir, listdir[i]),
                  allow_pickle=True)["arr_0"].item()
    gradient_syns[i], spect = synergy_bias(pid, lattice, layers)
    
    if gradient_syns[i] < min_syn and spect.max() < 0.5:
        min_syn = gradient_syns[i]
        min_syn_pid = pid
        min_syn_file = listdir[i]

min_syn_spect = synergy_bias(min_syn_pid, lattice, layers)[1]
min_syn_tpm = tensor[:,:,2]
#%%

choice_pids = [min_syn_pid, max_syn_pid]
tpms = [min_syn_tpm, max_syn_tpm]
palette = sns.diverging_palette(220, 20, n=7, center="dark")[::-1]

plt.figure(figsize=(8,4.5), dpi=250)
counter = 1
for pid in choice_pids:
            
    totals = {i : 0.0 for i in range(max(layers.values())+1)}
    mi = pid["{0:1:2}"][0]
    
    for i in range(max(layers.values())+1):
        totals[i] += sum({pid[x][1] for x in pid.keys() if layers[x] == i})/mi
    
    totals = {i : totals[i] / max(totals.values()) for i in totals.keys()}
    
    keys = np.array(list(lattice.nodes))
    vals = np.array([pid[key][1] for key in keys])
    vals = vals / vals.sum()
    
    #plt.figure(figsize=(12,12), frameon=False, dpi=250)
    plt.subplot(1,2,counter)
    
    plt.hlines(0, 0, 5, linestyle="--", color="k", linewidth=2, alpha=0.25)
    plt.hlines(-12, 0, 5, linestyle="--", color="k", linewidth=2, alpha=0.25)
    for i in range(2, 11, 2):
        plt.hlines(-1*i, -2, 5, linestyle="--", color="k", linewidth=2, alpha=0.25)
   
    nx.draw_networkx(lattice, 
                     pos, 
                     with_labels=False,
                     node_color="k",
                     node_size=500,
                     edge_color="k")
                     
    nx.draw_networkx(lattice, 
                     pos, 
                     with_labels=False,
                     node_color=(vals),
                     cmap="gray_r",
                     node_size=400,
                     edge_color="k",
                     width=2)
    
    plt.text(x = -1.6,
             y = -13.25, 
             s = "{0}{1}{2}",
             fontsize="large",)
             #fontweight="bold")

    plt.text(x = -0.9,
             y = 1, 
             s = "{012}", 
             fontsize="large",)
             #fontweight="bold")
    
    for i in range(7):
        plt.hlines(-2*i, 5, 
                   (5+((5**(totals[6-i]))-1)), 
                   linewidth=20, 
                   colors=palette[i])
    
    plt.vlines(x = 5, ymin = 1, ymax = -13, 
               color="k", linewidth=3)
    
    bias = synergy_bias(pid, lattice, layers)[0]
    plt.title("Synergy Bias: {0}".format(round(bias,3)))
    
    counter += 1
        
plt.tight_layout()
fig_dir = '/geode2/home/u100/tvarley/Carbonate/Libraries/PID_emergence/figure_scripts/final_figs/'
plt.savefig(fig_dir + "fig_1.svg", dpi=250, bbox_inches="tight")
plt.savefig(fig_dir + "fig_1.png", dpi=250, bbox_inches="tight")
