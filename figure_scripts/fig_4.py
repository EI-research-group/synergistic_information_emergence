#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:13:40 2021

@author: thosvarley
"""

import os
import numpy as np 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt
import networkx as nx 
import seaborn as sns

in_dir = os.getcwd()
out_dir = in_dir + "/figure_scripts/final_figs/"

lattice_3 = nx.read_gml("lattice_3.gml")
lattice_4 = nx.read_gml("lattice_4.gml")
lattice_5 = nx.read_gml("lattice_5.gml")

layers_3 = nx.shortest_path_length(lattice_3, source="{0}{1}{2}")
layers_4 = nx.shortest_path_length(lattice_4, source="{0}{1}{2}{3}")
layers_5 = nx.shortest_path_length(lattice_5, source="{0}{1}{2}{3}{4}")

#%%

## XOR GATE
micro = np.load(in_dir + '/XOR/XOR_micro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]
macro = np.load(in_dir + '/XOR/XOR_macro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]

micro_spectrum = np.zeros(max(layers_5.values())+1)
macro_spectrum = np.zeros(max(layers_3.values())+1)

macro_mi = macro["{0:1:2}"][0]
micro_mi = micro["{0:1:2:3:4}"][0]

for i in range(micro_spectrum.shape[0]):
    micro_spectrum[i] += sum([micro[key][1] for key in micro.keys() if layers_5[key] == i]) / micro_mi
for i in range(macro_spectrum.shape[0]):
    macro_spectrum[i] += sum([macro[key][1] for key in macro.keys() if layers_3[key] == i]) / macro_mi

#%%
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.bar(np.arange(micro_spectrum.shape[0]), 
        micro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(micro_spectrum), center="dark"))
plt.ylabel("Proportion of PID")
plt.title("Micro-Scale")
plt.xlim([0, len(micro_spectrum)])
plt.ylim([0,1.05])

plt.subplot(1,2,2)
plt.bar(np.arange(macro_spectrum.shape[0]), 
        macro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(macro_spectrum), center="dark"))
plt.ylabel("Proportion of PID")
plt.title("Macro-Scale")
plt.xlim([0, len(macro_spectrum)])
plt.ylim([0,1.05])

plt.tight_layout()
plt.savefig(out_dir + "xor_spectra.png", dpi=250, bbox_inches="tight")
plt.show()

## AND GATE
#%%

micro = np.load(in_dir + '/AND/AND_micro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]
macro = np.load(in_dir + '/AND/AND_macro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]

micro_spectrum = np.zeros(max(layers_4.values()))
macro_spectrum = np.zeros(max(layers_3.values()))

macro_mi = macro["{0:1:2}"][0]
micro_mi = micro["{0:1:2:3}"][0]

for i in range(micro_spectrum.shape[0]):
    micro_spectrum[i] += sum([micro[key][1] for key in micro.keys() if layers_4[key] == i]) / micro_mi
for i in range(macro_spectrum.shape[0]):
    macro_spectrum[i] += sum([macro[key][1] for key in macro.keys() if layers_3[key] == i]) / macro_mi

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.bar(np.arange(micro_spectrum.shape[0]), 
        micro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(micro_spectrum), center="dark"))
plt.ylabel("Proportion of PID")
plt.title("Micro-Scale")
plt.xlim([0, len(micro_spectrum)])
plt.ylim([0,0.8])

plt.subplot(1,2,2)
plt.bar(np.arange(macro_spectrum.shape[0]), 
        macro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(macro_spectrum), center="dark"))
plt.ylabel("Proportion of PID")
plt.title("Macro-Scale")
plt.xlim([0, len(macro_spectrum)+1])
plt.ylim([0,0.8])

plt.tight_layout()
plt.savefig(out_dir + "and_spectra.png", dpi=250, bbox_inches="tight")

plt.show()

## OR GATE
#%%

micro = np.load(in_dir + '/OR/OR_micro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]
macro = np.load(in_dir + '/OR/OR_macro_PID_dicts.npz',
                    allow_pickle=True)["arr_0"][1]

micro_spectrum = np.zeros(max(layers_5.values())+1)
macro_spectrum = np.zeros(max(layers_3.values())+1)

macro_mi = macro["{0:1:2}"][0]
micro_mi = micro["{0:1:2:3:4}"][0]

for i in range(micro_spectrum.shape[0]):
    micro_spectrum[i] += sum([micro[key][1] for key in micro.keys() if layers_5[key] == i]) / micro_mi
for i in range(macro_spectrum.shape[0]):
    macro_spectrum[i] += sum([macro[key][1] for key in macro.keys() if layers_3[key] == i]) / macro_mi

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.bar(np.arange(micro_spectrum.shape[0]), 
        micro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(micro_spectrum), center="dark"))
plt.xlabel("PID Lattice Layers")
plt.ylabel("Proportion of PID")
plt.title("Micro-Scale")
plt.xlim([0, len(micro_spectrum)])
plt.ylim([0,0.8])

plt.subplot(1,2,2)
plt.bar(np.arange(macro_spectrum.shape[0]), 
        macro_spectrum,
        color=sns.diverging_palette(250, 15, s=75, l=40, n=len(macro_spectrum), center="dark"))
plt.xlabel("PID Lattice Layers")
plt.ylabel("Proportion of PID")
plt.title("Macro-Scale")
plt.xlim([0, len(macro_spectrum)])
plt.ylim([0,0.8])

plt.tight_layout()
plt.savefig(out_dir + "or_spectra.png", dpi=250, bbox_inches="tight")
plt.show()

#%%
