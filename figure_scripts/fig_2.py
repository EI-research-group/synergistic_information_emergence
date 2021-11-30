#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:40:47 2021

@author: tvarley
"""

import os
import numpy as np 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

in_dir = os.getcwd() + "data/gradient/"

tensor_3 = np.load(gradient_dir + "/tensor_3.npz")["arr_0"]
tensor_4 = np.load(gradient_dir + "/tensor_4.npz")["arr_0"]
tensor_5 = np.load(gradient_dir + "/tensor_5.npz")["arr_0"]

tensors = [tensor_3, tensor_4, tensor_5]

pal = sns.color_palette("YlGn", n_colors=3)[::-1]

G_3 = nx.complete_graph(3, create_using=nx.DiGraph())
G_3 = nx.relabel_nodes(G_3, {0:r"$A$", 
                             1:r"$B$", 
                             2:r"$C$"})
pos_3 = nx.circular_layout(G_3)

G_4 = nx.complete_graph(4, create_using=nx.DiGraph())
G_4 = nx.relabel_nodes(G_4, {0:r'$\alpha$', 
                             1:r'$\beta$', 
                             2:r'$B$', 
                             3:r'$C$'})
pos_4 = nx.circular_layout(G_4)
pos_4 = {key : pos_4[key] + np.array([3, 0]) for key in pos_4.keys()}

G_5 = nx.complete_graph(5, create_using=nx.DiGraph())
G_5 = nx.relabel_nodes(G_5, {0:r'$\alpha_0$', 
                             1:r'$\alpha_1$', 
                             2:r'$\beta$', 
                             3:r'$B$',
                             4:r'$C$'})
pos_5 = nx.circular_layout(G_5)
pos_5 = {key : pos_5[key] + np.array([6, 0]) for key in pos_5.keys()}

Gs = [G_3, G_4, G_5]

colors = [
    [pal[0] for x in range(len(G_3))],
    [pal[1], pal[1], pal[0], pal[0]],
    [pal[2], pal[2], pal[1], pal[0], pal[0]]
    ]

fig = plt.figure(figsize=(9,6.5), dpi=250, frameon=True)
for i in range(len(Gs)):
    ax = plt.subplot(2,3,i+1)
    pos = nx.circular_layout(Gs[i])

    nx.draw_networkx(Gs[i], pos, 
                     with_labels=False,
                     node_size=10**3.05, 
                     node_color="k")

    nx.draw_networkx(Gs[i], pos, 
                     node_size=10**3, 
                     width=3,
                     node_color=colors[i],
                     font_size=20,
                     font_weight="bold")
    
    if i == 0:
        c = mpatches.Circle(xy = pos["$A$"], 
                            radius = 0.25,
                            edgecolor="cornflowerblue",
                            facecolor="cornflowerblue",
                            fill=True,
                            alpha=0.25,
                            linewidth=2)
        ax.add_patch(c)
        plt.title("Macro-Scale System")
        
    elif i == 1:
        c = mpatches.Ellipse(xy = (pos['$\\alpha$']+pos['$\\beta$'])/2,
                            width=0.65, 
                            height=2,
                            angle=45,
                            edgecolor="cornflowerblue",
                            facecolor="cornflowerblue",
                            fill=True,
                            alpha=0.25,
                            linewidth=2)
        
        c1 = mpatches.Circle(xy = pos['$\\alpha$'],
                             radius = 0.3,
                             edgecolor="firebrick",
                             facecolor="firebrick",
                             fill=True,
                             alpha=0.25,
                             linewidth=2)
        ax.add_patch(c)
        ax.add_patch(c1)
        plt.title("Meso-Scale System")
    else:
        c = mpatches.Ellipse(xy = (pos['$\\alpha_0$']+pos['$\\alpha_1$'])/2,
                            width=0.6, 
                            height=1.65,
                            angle=35,
                            edgecolor="firebrick",
                            facecolor="firebrick",
                            fill=True,
                            alpha=0.25,
                            linewidth=2)
        ax.add_patch(c)
        plt.title("Micro-Scale System")

for i in range(len(tensors)):
    ax2 = plt.subplot(2,3,i+4)
    plt.imshow(tensors[i][:,:,3], vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.ylabel("Time" + r" $t-1$")
    plt.xlabel("Time" + r" $t$")
    
    if i == 0:
        plt.title("Macro-Scale TPM")
    elif i == 1:
        plt.title("Meso-Scale TPM")
    else:
        plt.title("Micro-Scale TPM")

plt.tight_layout()
fig_dir = in_dir + "/figure_scripts/final_figs/"
plt.savefig(fig_dir + "fig_2.svg", dpi=250, bbox_inches="tight")
plt.savefig(fig_dir + "fig_2.pdf", dpi=250, bbox_inches="tight")
