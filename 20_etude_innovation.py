#!/usr/bin/python
# -*- coding: UTF-8 -*-
""" *Optimisation* du déplacement d'un véhicule propulsé électriquement,
appliquée au Challenge ÉducEco

Ce petit script sert à étudier différentes façons 
de générer des profils de vitesse de façon aléatoire,
pour alimenter l'optimisation par recherche aléatoire :
 * génération "bruit gaussien AR(1)"
 * génération "coup de pédale aléatoire"

Pierre Haessig — Décembre 2011
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from optim_circuit import l, v0, v_min, random_v, random_v2


plt.figure('bruit AR(1)')
n = 5
for i in range(n):
    plt.plot(l,random_v(v0)[0], color=(0,0.5,i/(n-1)))
plt.hlines([v_min-0.004,v_min+0.004], *plt.xlim(),
           linestyles='dashed', colors='blue')
plt.title('Bruit AR(1), avec amplitude Rayleigh')
plt.xlabel('abscisse l [m]')
plt.ylabel('vitesse [m/s]')
plt.grid(True)

plt.figure('bruit boost')
n = 3
for i in range(n):
    plt.plot(l,random_v2(v0)[0], color=(0,0.5,i/(n-1)))
#plt.hlines([v_min-0.004,v_min+0.004], *plt.xlim(),
#           linestyles='dashed', colors='blue')
plt.title('Bruit "boost accel", avec amplitude Rayleigh')
plt.xlabel('abscisse l [m]')
plt.ylabel('vitesse [m/s]')
plt.grid(True)

plt.show()
