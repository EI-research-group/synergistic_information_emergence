#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:22:40 2021

@author: tvarley
"""

import os 
import sys
from dit.pid import PID_WB
import numpy as np 
from library import equilibrium_dist, make_microscale, pid_dicts, dit_dist

in_dir = os.getcwd()

if os.path.isdir(in_dir + "/data/discrete/") == False:
    os.mkdir(in_dir + "/data/")
    os.mkdir(in_dir + "/data/discrete")
    os.mkdir(in_dir + "/data/discrete/mat_3/")
    os.mkdir(in_dir + "/data/discrete/pid_3/")

    os.mkdir(in_dir + "/data/discrete/mat_4/")
    os.mkdir(in_dir + "/data/discrete/pid_4/")

    os.mkdir(in_dir + "/data/discrete/mat_5/")
    os.mkdir(in_dir + "/data/discrete/pid_5/")

out_dir = in_dir + "/data/discrete/"

idx = int(sys.argv[1])
bias = 0.01

mapping = {
    "00" : "0",
    "01" : "0",
    "10" : "0",
    "11" : "1"}

choices = np.load(in_dir + "choice_files.npz")["arr_0"]
choice = choices[idx][:-4]

inds = [int(x) for x in choice]
mat_3 = np.zeros((len(choice), len(choice)))
mat_3[np.arange(len(choice)), inds] = 1 - bias
mat_3[mat_3 == 0] = bias/(mat_3.shape[0]-1)

mat_4 = make_microscale(mat=mat_3, split=0, mapping=mapping)
mat_5 = make_microscale(mat=mat_4, split=0, mapping=mapping)

eq_3 = equilibrium_dist(mat_3)
eq_3 = np.repeat(1/mat_3.shape[0], mat_3.shape[0])
dist_3 = dit_dist(eq_3, mat_3)
imin_3 = PID_WB(dist_3)
arr_3 = np.array([pid_dicts(imin_3)[1]], dtype="object")
np.savez_compressed(out_dir + "pid_3/{0}.npz".format(str(idx)), arr_3)
np.savez_compressed(out_dir + "mat_3/{0}.npz".format(str(idx)), mat_3)

eq_4 = equilibrium_dist(mat_4)
eq_4 = np.repeat(1/mat_4.shape[0], mat_4.shape[0])
dist_4 = dit_dist(eq_4, mat_4)
imin_4 = PID_WB(dist_4)
arr_4 = np.array([pid_dicts(imin_4)[1]], dtype="object")
np.savez_compressed(out_dir + "pid_4/{0}.npz".format(str(idx)), arr_4)
np.savez_compressed(out_dir + "mat_4/{0}.npz".format(str(idx)), mat_4)

eq_5 = equilibrium_dist(mat_5)
eq_5 = np.repeat(1/mat_5.shape[0], mat_5.shape[0])
dist_5 = dit_dist(eq_5, mat_5)
imin_5 = PID_WB(dist_5)
arr_5 = np.array([pid_dicts(imin_5)[1]], dtype="object")
np.savez_compressed(out_dir + "pid_5/{0}.npz".format(expansion, str(idx)), arr_5)
np.savez_compressed(out_dir + "mat_5/{0}.npz".format(expansion, str(idx)), mat_5)

