#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:11:28 2021

@author: tvarley
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import linregress, entropy

in_dir = os.getcwd()

def syn_bias(spectrum):
    dist = np.arange(spectrum.shape[0])
    return (dist * spectrum).sum() / (dist.max())

def red_bias(spectrum):
    dist = np.arange(spectrum.shape[0])[::-1]
    return (dist * spectrum).sum() / (dist.max())

def determinism(mat):
    ents = 0.0
    for i in range(mat.shape[0]):
        ents += entropy(mat[i], base=2)
    ents /= mat.shape[0]
    return (np.log2(mat.shape[0])-ents)/np.log2(mat.shape[0])

lattice_3 = nx.read_gml(in_dir + "lattice_3.gml")
lattice_4 = nx.read_gml(in_dir + "lattice_4.gml")
lattice_5 = nx.read_gml(in_dir + "lattice_5.gml")

layers_3 = nx.shortest_path_length(lattice_3, source="{0}{1}{2}")
layers_4 = nx.shortest_path_length(lattice_4, source="{0}{1}{2}{3}")
layers_5 = nx.shortest_path_length(lattice_5, source="{0}{1}{2}{3}{4}")

#%%
#The Gaussian Matrices 

gaussian_dir = in_dir + "data/gaussian/"
listdir = os.listdir(gaussian_dir + "pid_5/")
g_syns_3 = np.zeros(len(listdir))
g_syns_5 = np.zeros(len(listdir))

counter = 0
for file in listdir:
    
    pid_3 = np.load(gaussian_dir + "pid_3/{0}".format(file), allow_pickle=True)["arr_0"].item()
    pid_5 = np.load(gaussian_dir + "pid_5/{0}".format(file), allow_pickle=True)["arr_0"].item()
    
    mi_3 = pid_3["{0:1:2}"][0]
    mi_5 = pid_5["{0:1:2:3:4}"][0]
    
    totals_3 = {i : 0.0 for i in range(max(layers_3.values())+1)}
    totals_5 = {i : 0.0 for i in range(max(layers_5.values())+1)}
    
    for i in range(max(layers_5.values())+1):
        totals_5[i] = sum({pid_5[x][1] for x in pid_5.keys() if layers_5[x] == i})/mi_5
        if i < max(layers_3.values())+1:
                    totals_3[i] = sum({pid_3[x][1] for x in pid_3.keys() if layers_3[x] == i})/mi_3

    spectrum_3 = np.array(list(totals_3.values()))
    spectrum_5 = np.array(list(totals_5.values()))
    
    g_syns_3[counter] = syn_bias(spectrum_3)
    g_syns_5[counter] = syn_bias(spectrum_5)
    
    counter += 1

#%%
#The pure Gaussian and pure Deterministic

selected_dir = in_dir + "data/selected/"

listdir = os.listdir(selected_dir + "pid_5/")

s_syns_3 = np.zeros(len(listdir))
s_syns_5 = np.zeros(len(listdir))

counter = 0
for file in listdir:
    
    pid_3 = np.load(selected_dir + "pid_3/{0}".format(file), allow_pickle=True)["arr_0"].item()
    pid_5 = np.load(selected_dir + "pid_5/{0}".format(file), allow_pickle=True)["arr_0"].item()
    
    mi_3 = pid_3["{0:1:2}"][0]
    mi_5 = pid_5["{0:1:2:3:4}"][0]
    
    totals_3 = {i : 0.0 for i in range(max(layers_3.values())+1)}
    totals_5 = {i : 0.0 for i in range(max(layers_5.values())+1)}
    
    for i in range(max(layers_5.values())+1):
        totals_5[i] = sum({pid_5[x][1] for x in pid_5.keys() if layers_5[x] == i})/mi_5
        if i < max(layers_3.values())+1:
                    totals_3[i] = sum({pid_3[x][1] for x in pid_3.keys() if layers_3[x] == i})/mi_3

    spectrum_3 = np.array(list(totals_3.values()))
    spectrum_5 = np.array(list(totals_5.values()))
    
    s_syns_3[counter] = syn_bias(spectrum_3)
    s_syns_5[counter] = syn_bias(spectrum_5)
    
    counter += 1

#%%

fig_dir = in_dir + "/figure_scripts/final_figs/"

magma = sns.color_palette("magma", n_colors=10)

plt.figure(figsize=(8,4), dpi=250)

plt.subplot(1,2,1)

xspace = np.linspace(g_syns_3.min(), g_syns_3.max())
lr = linregress(g_syns_3,
                g_syns_3 - g_syns_5)
ypred = lr[1]+(lr[0]*xspace)

plt.scatter(g_syns_3,
            g_syns_3 - g_syns_5,
            s=20,
            color=magma[2],
            label="Gaussian Systems",
            alpha=0.5)

plt.plot(xspace, 
         ypred, 
         linestyle="--", 
         linewidth=2.5,
         color="grey",
         label="r = {0}\np < 10e-10".format(round(lr[2],3)))

plt.title("Gaussian Systems")
plt.xlabel("Macro-Scale Synergy Bias")
plt.ylabel("Change in Synergy Bias")
plt.legend()


plt.subplot(1,2,2)

plt.scatter(g_syns_3,
            g_syns_3 - g_syns_5,
            s=20,
            color=magma[2], 
            alpha=0.5, 
            label="Gaussian")

plt.scatter(s_syns_3,
            s_syns_3 - s_syns_5,
            s=20,
            color=magma[-2],
            alpha=0.5,
            label="Deterministic")

plt.vlines(0.5,
           (s_syns_3-s_syns_5).min(),
           (g_syns_3-g_syns_5).max(),
           linestyle="--",
           color="grey")
plt.hlines(0,
           s_syns_3.min(),
           g_syns_3.max(),
           linestyle="--",
           color="grey")

plt.title("Gaussian & Deterministic")
plt.xlabel("Macro-Scale Synergy Bias")
plt.ylabel("Change in Synergy Bias")
plt.legend()

plt.tight_layout()
plt.savefig(fig_dir + "fig_3.svg", dpi=250)
plt.savefig(fig_dir + "fig_3.pdf", dpi=250)
plt.show()

