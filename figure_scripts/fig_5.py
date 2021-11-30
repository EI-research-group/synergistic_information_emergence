#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:57:28 2021

@author: tvarley
"""

import os
import numpy as np 
from scipy.stats import entropy 
import matplotlib.pyplot as plt

cwd = in_dir = os.getcwd()
fig_dir = cwd + "/figure_scripts/final_figs/"

def negentropy(X):
    return (np.log2(len(X)) - entropy(X, base=2))

def determinism(mat):
    ents = []
    for i in range(mat.shape[0]):
        ents.append(entropy(mat[i], base=2))
    return np.log2(mat.shape[0]) - np.mean(ents)

def degeneracy(mat):
    avg = np.mean(mat, axis=0)
    return (np.log2(mat.shape[0]) - entropy(avg, base=2))

in_dir = cwd + "/data/discrete/"
listdir = os.listdir(in_dir + "mat_3")

ents_3 = np.zeros(len(listdir))
negs_3 = np.zeros(len(listdir))
dets_3 = np.zeros(len(listdir))
degs_3 = np.zeros(len(listdir))
joint_3 = np.zeros(len(listdir))
avg_outs_3 = np.zeros(len(listdir))

ents_5 = np.zeros(len(listdir))
negs_5 = np.zeros(len(listdir))
dets_5 = np.zeros(len(listdir))
degs_5 = np.zeros(len(listdir))
joint_5 = np.zeros(len(listdir))
avg_outs_5 = np.zeros(len(listdir))


counter = 0
for i in [x.split(".")[0] for x in listdir]:
    
    mat_3 = np.load(in_dir + "mat_3/{0}.npz".format(i))["arr_0"]
    mat_5 = np.load(in_dir + "mat_5/{0}.npz".format(i))["arr_0"]
    
    mat_3_joint = mat_3 / mat_3.sum()
    mat_5_joint = mat_5 / mat_5.sum()
    
    joint_3[counter] = entropy(mat_3_joint.flatten(), base=2)
    joint_5[counter] = entropy(mat_5_joint.flatten(), base=2)
    
    dets_3[counter] = determinism(mat_3)
    dets_5[counter] = determinism(mat_5)
    
    degs_3[counter] = degeneracy(mat_3)
    degs_5[counter] = degeneracy(mat_5)
    
    n_3 = 0.0
    e_3 = 0.0
    for j in range(mat_3.shape[0]):
        n_3 += negentropy(mat_3[j])
        e_3 += entropy(mat_3[j], base=2)
    
    n_5 = 0.0
    e_5 = 0.0
    for j in range(mat_5.shape[0]):
        n_5 += negentropy(mat_5[j])
        e_5 += entropy(mat_5[j], base=2)
    
    negs_3[counter] = n_3
    negs_5[counter] = n_5
    
    ents_3[counter] = e_3 / mat_3.shape[0]
    ents_5[counter] = e_5 / mat_5.shape[0]
    
    avg_outs_3[counter] = entropy(np.mean(mat_3, axis=0), base=2)
    avg_outs_5[counter] = entropy(np.mean(mat_5, axis=0), base=2)
    
    if counter < 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(mat_3)
        plt.colorbar(label="Probability",fraction=0.046, pad=0.04)
        plt.subplot(1,2,2)
        
        plt.imshow(mat_5)
        plt.colorbar(label="Probability",fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fig_dir + "deterministic_tpm.png", dpi=250, bbox_inches="tight")
        plt.show()
    
    counter += 1
    
#%%
eis_3 = dets_3 - degs_3
eis_5 = dets_5 - degs_5

change_ei = eis_3 - eis_5
change_det = dets_3 - dets_5
change_deg = degs_3 - degs_5
change_ent = ents_3 - ents_5
change_out = avg_outs_3 - avg_outs_5

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(change_ent, change_ei, s=20, alpha=0.5)
plt.xlabel(r"Change in $\langle H(X_{t} \to X_{t+1}) \rangle$")
plt.ylabel("Change in EI")
#plt.title("Determinism & EI Across Scales")
plt.vlines(0, change_ei.min(), change_ei.max(),
           linestyle="--",
           color="grey")
plt.hlines(0, change_ent.min(), change_ent.max(),
           linestyle="--",
           color="grey")

plt.subplot(1,3,2)
plt.scatter(change_out, change_ei, s=20, alpha=0.5)
plt.xlabel(r"Change in $H(\langle X_{t} \to X_{t+1} \ranlge)$")
plt.ylabel("Change in EI")
#plt.title("Degeneracy & EI Across Scales")
plt.vlines(0, change_ei.min(), change_ei.max(),
           linestyle="--",
           color="grey")
plt.hlines(0, change_out.min(), change_out.max(),
           linestyle="--",
           color="grey")

plt.subplot(1,3,3)
plt.scatter(change_det - change_deg, change_ei, s=20, alpha=0.5)
plt.xlabel("Change in Determinism - Change in Degeneracy")
plt.ylabel("Change in EI")
#plt.title("EI Across Scales")
plt.hlines(0, (change_det - change_deg).min(), 
           (change_det - change_deg).max(),
           linestyle="--",
           color="grey")
plt.vlines(0, change_ei.min(), change_ei.max(),
           linestyle="--",
           color="grey")

plt.tight_layout()
plt.savefig(fig_dir + "fig_5.png", dpi=250, bbox_inches="tight")
plt.savefig(fig_dir + "fig_5.svg", dpi=250, bbox_inches="tight")
