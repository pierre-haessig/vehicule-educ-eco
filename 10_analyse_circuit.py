#!/usr/bin/python
# -*- coding: UTF-8 -*-
""" Analyse du déplacement d'un véhicule propulsé électriquement,
appliquée au Challenge ÉducEco

Dans cette analyse simple la vitesse de déplacement 
est maintenue constante

Pierre Haessig — Décembre 2011
"""

from __future__ import division, print_function
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Paramètres du circuit
dossier_circuit = 'Circuit_Nogaro'
L = 3636.0 # [m]
N = 3637*1 # nombre de points de discrétisation
# Vitesse moyenne minimale (30 km/h)
v_min = 30/3.6 # [m/s]

# Paramètres véhicule:
m = 200 #[kg]
S = 1.05 # [m²]
Cx = 0.5 # +/- 0.1 [N/...]
# Coeff d'adhérence pneu-route
Cpr = 0.01 # [N/N] pneu standard
Cpr = 0.0015 # [N/S] pneu spécial Éco

# Coefficient de pertes Joules (Pj = Cj*f_mot²)
Cj = 100./50**2 # [W/N²] (100 W perdus à 50 N)

####################################################################
# Discrétisation du circuit
dl = L/(N-1)
l = np.linspace(0,L,N)
print('Circuit de %.1f m discrétisé avec un pas de %.1f m [%d points]' 
      % (L,dl,N))

# Altitude z(l)
z_data = np.loadtxt(os.path.join(dossier_circuit,'altitude.csv'), 
                    delimiter=',', skiprows=4)
altitude = interp1d(x=z_data[:,0], y=z_data[:,1], kind='linear')

z = altitude(l)
dzdl = np.gradient(z,dl)

# Vitesse constante:
v = np.ones(N)*v_min
v = np.load('vitesse_best_0977.npy')
profil_v = 'best_0977'
dvdl = np.gradient(v,dl)

# Calcul des forces:
g = 9.81 # [m/s²]
SCx = S*Cx
f_pr = m*g*Cpr * np.ones(N)
f_aero = SCx*v**2
f_pes = m*g*dzdl
f_iner = m*v*dvdl

# Force que doit fournir le moteur :
f_mot = f_iner + f_pr + f_aero + f_pes

### Calcul des puissances :
P_mot = f_mot*v
P_joule = Cj*f_mot**2

### Calcul des énergies dépensées:
E_mot = f_mot.sum()*dl
E_joule = (P_joule/v).sum()*dl
E_totale = E_mot + E_joule

### Bilan écrit #########################

print('Bilan des forces :')
print(' - Résistance au roulement : %.1f N (constant)' % f_pr[0])
print(' - Résistance aérodynamique : %.1f N en moyenne' %
      (f_aero.mean()) )
print(' - Pesanteur : %.1f N max' % (np.abs(f_pes)).max() )
print(' - "Force" inertielle : %.1f N max' % (np.abs(f_iner)).max() )

print('\nBilan d\'énergie sur le tour:')
print('1) Énergie consommée mécaniquement : %.1f kJ' % 
      (E_mot/1000) )
print('dont : ')
print('  - résistance au roulement : %.1f kJ' %
      (f_pr.sum()*dl/1000))
print('  - trainée aérodynamique : %.1f kJ' %
      (f_aero.sum()*dl/1000))
print('  - variation d\'énergie cinétique : %.1f kJ' %
      (f_iner.sum()*dl/1000))
print('2) Pertes Joule dans le moteur : %.1f kJ' % 
      (E_joule/1000) )
print('Énergie consommée au total: %.1f kJ' % 
      (E_totale/1000) )

#####################################################################
# Tracés:
fig = plt.figure('Analyse de %s' % dossier_circuit )
ax=fig.add_subplot(211, title='altitude & pente "%s"' % dossier_circuit,
                   ylabel='alitude [m]')
ax.plot(z_data[:,0], z_data[:,1], 'bd')
ax.plot(l, z, 'b')
ax.grid(True)
ax.legend(('mesures', 'interpol'), loc='upper left')
ax = ax.twinx() # plot superposé avec des unités différentes
ax.plot(l, dzdl*100, 'g')
ax.plot(l, dzdl*0, 'g--')
ax.set_ylabel('pente (%)')
#ax.grid(True)
ax.legend(('pente',), loc='upper right')


ax=fig.add_subplot(212, title=u'Bilan des forces (profil vitesse "%s")' % profil_v,
                   xlabel='abscisse l [m]',
                   ylabel='force [N]', sharex=ax)
ax.grid(True)
ax.plot(l, f_mot, 'red', linewidth=2, 
        label='total')
ax.plot(l, f_pes, 'blue', 
        label='pesanteur')
ax.plot(l, f_aero, 'green', 
        label=u'trainée aéro.')
ax.plot(l, f_pr, 'cyan',
        label='roulement')
ax.plot(l, f_iner, 'orange',
        label='inertie')
ax.legend()

### Bilan de puissance:
fig = plt.figure('Bilan puissance de %s' % dossier_circuit)
ax=fig.add_subplot(111, title='Bilan de puissance "%s"' %\
                               dossier_circuit,
                        xlabel='abscisse l [m]',
                        ylabel='Puissance [W]')
ax.plot(l, P_mot, 'blue', label='$f_{mot} \cdot v(l)$')
ax.plot(l, P_joule, 'red', label='dissipation Joule')
ax.grid(True)
ax.legend()


plt.show()
