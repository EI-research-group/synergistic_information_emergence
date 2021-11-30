#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 4 09:05:40 2021

@author: tvarley
"""

import os
import sys
from dit.pid import PID_WB
import numpy as np
from library import equilibrium_dist, make_microscale, pid_dicts, dit_dist

idx = int(sys.argv[1])

in_dir = os.getcwd() + "/data/gaussian/" 
out_dir = in_dir

tensor = np.load(in_dir + "tensor_3.npz")["arr_0"]
mat_3 = tensor[:,:,idx]

mapping = {
    "00" : "0",
    "01" : "0",
    "10" : "0",
    "11" : "1"}

mat_4 = make_microscale(mat=mat_3, split=0, mapping=mapping)
mat_5 = make_microscale(mat=mat_4, split=0, mapping=mapping)
	
eq_4 = equilibrium_dist(mat_4)
dist_4 = dit_dist(eq_4, mat_4)
imin_4 = PID_WB(dist_4)
arr_4 = np.array([pid_dicts(imin_4)[1]], dtype="object")
np.savez_compressed(out_dir + "pid_4/{0}.npz".format(str(idx)), arr_4)
np.savez_compressed(out_dir + "mat_4/{0}.npz".format(str(idx)), mat_4)

eq_5 = equilibrium_dist(mat_5)
dist_5 = dit_dist(eq_5, mat_5)
imin_5 = PID_WB(dist_5)
arr_5 = np.array([pid_dicts(imin_5)[1]], dtype="object")
np.savez_compressed(out_dir + "pid_5/{0}.npz".format(str(idx)), arr_5)
np.savez_compressed(out_dir + "mat_5/{0}.npz".format(str(idx)), mat_5)

