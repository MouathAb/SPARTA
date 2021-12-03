#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:48:57 2021

@author: amouath
"""
import numpy as np

a = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
b = np.array([1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0])
c = b.copy()
dis = 0
pos = 0
ici = []
for n,i in enumerate(b):
    
    if i != 1:
        
        dis += 1
        pos = n - dis + 1
    else:
        print(pos,dis)
        if pos != 0 :
            ici += [(dis,pos)]
        dis = 0
        pos = 0

for d,p in ici:
    if d < 3:
        c[p:p+d] = [1]*d
        

dis = 0
pos = 0
labas = []
for n,i in enumerate(c):
    if i != 0:
        dis += 1
        pos = n - dis + 1
    else:
        if dis != 0:
            labas += [(dis,pos)]
        dis = 0
        pos = 0

for d,p in labas:
    if d < 3:
        c[p:p+d] = [0]*d