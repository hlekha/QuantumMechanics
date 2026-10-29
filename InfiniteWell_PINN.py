#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytorch
import math
import numpy as np
from numpy import sin, pi
from scipy import constants

def actual_solution(x, t, L):
    hbar = constants.hbar
    theta = n*pi*x/L
    i = 1j
    e = (-i*E*t)/hbar
    return math.sqrt(2/L)*sin(theta)*math.exp(e)

def boundary(x, L):
    if x <= 0 or x >= L:
        raise ValueError("outside of box")

