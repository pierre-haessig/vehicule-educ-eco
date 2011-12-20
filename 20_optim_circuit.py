#!/usr/bin/python
# -*- coding: UTF-8 -*-
""" *Optimisation* du déplacement d'un véhicule propulsé électriquement,
appliquée au Challenge ÉducEco

Le but est de trouver le meilleur profil de vitesse pour minimiser
l'énergie totale dépensée sur un tour

Pierre Haessig — Décembre 2011
"""

from __future__ import division, print_function
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.interpolate import interp1d
#from scipy.optimize import fmin_bfgs

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

# Coefficients dérivés et constantes:
g = 9.81 # [m/s²]
SCx = S*Cx

####################################################################
# Discrétisation du circuit
dl = L/(N-1)
l = np.linspace(0,L,N)


# Altitude z(l)
z_data = np.loadtxt(os.path.join(dossier_circuit,'altitude.csv'), 
                    delimiter=',', skiprows=4)
altitude = interp1d(x=z_data[:,0], y=z_data[:,1], kind='linear')

z = altitude(l)
dzdl = np.gradient(z,dl)



def u2v(u, v_moy):
    '''conversion des variables u à v
    u : vecteur écarts *libres* (dim N-1)
    v_moy : vitesse moyenne
    
    Renvoie
    -------
    v : vitesses (dim N)
    
    Formule appliquée :
    v(0)   = v_moy + u(0)
    v(i)   = v_moy + u(i) - u(i-1) pour i in [1, N-2]
    v(N-1) = v_moy +      - u(N-2)
    
    On peut voir u(i) comme un écart de position
    par rapport à la position qu'il faut avoir pour
    que la vitesse soit constante égale à v_moy
    '''
    v = np.ones(N)*v_moy
    v[0:N-1] += u
    v[1:N]   -= u
    return v
     
# Vitesse constante:
u0 = np.zeros(N-1)
v0 = u2v(u0, v_min)

def crit_E_tot(v, with_penalty=True):
    '''critère à optimiser : 
    Énergie totale consommée par le moteur électrique,
    fonction du profil de vitesse v(l)
    
    pour lever la contrainte sur la moyenne de la vitesse v,
    on utilise comme variable le vecteur u de dimension N-1
    '''
    dvdl = np.gradient(v,dl)
    # Calcul des forces:
    f_pr = m*g*Cpr * np.ones(N)
    f_aero = SCx*v**2
    f_pes = m*g*dzdl
    f_iner = m*v*dvdl
    # Force que doit fournir le moteur :
    f_mot = f_iner + f_pr + f_aero + f_pes    
    ### Calcul des énergies dépensées:
    E_mot = f_mot.sum()*dl
    E_joule = (Cj*f_mot**2/v).sum()*dl
    E_totale = E_mot + E_joule
    # Décompte de la perte d'énergie cinétique en fin de tour
    #dE_cin = m/2.*(v[-1]**2 - (v[0]**2))
    penalty = 0
    if with_penalty:
        # pénalise une différence de vitesse entre 
        # le début et la fin du tour
        penalty = 10**5 * (v[0]-v[-1])**2
    return (E_totale + penalty)

E0 = crit_E_tot(v0)
print("Énergie consommée à vitesse constante : %.1f kJ" %
      (E0/1000))

def random_v(v_prev):
    '''génère un vecteur vitesse aléatoirement
    en se basant sur un vecteur vitesse dont la moyenne
    sera inchangée'''
    N = len(v_prev)
    scale = np.random.rayleigh(0.004)
    phi = 1-10**np.random.uniform(-8,-2)
    # phi = 0.99999
    #scale = 0.1
    noise = np.random.normal(size=N, scale=scale)
    # Filtrage passe-bas
    noise = lfilter([np.sqrt(1-phi**2)],[1. , -phi], noise)
    #noise = lfilter([1/np.sqrt(N)],[1. , -1], noise)
    v = v_prev + (noise - noise.mean())
    return v, scale
    
def random_search(niter):
    '''recherche aléatoire d'un minimiseur'''
    np.random.seed(0)
    v0 = np.ones(N)*v_min
    E = crit_E_tot(v0)
    v_best = v0
    E_best = E
    E_list = []
    i_list = []
    scale_list = []
    for i in xrange(niter):
        v,scale = random_v(v_best)
        E = crit_E_tot(v)
        
        if E < E_best:
            # Sauvegarder le progrès
            E_best = E
            v_best = v
            E_list.append(crit_E_tot(v, with_penalty=False))
            i_list.append(i)
            scale_list.append(scale)
    
    return (v_best, E_best, E_list, i_list, scale_list)

n_iter = 10**5
print('Optimisation aléatoire en %d itérations...' %n_iter)
result = random_search(n_iter)
v_best, E_best, E_list, i_list, scale_list = result
E_best = crit_E_tot(v_best, with_penalty=False)
print('itérations apportant une amélioration : %d' % len(E_list))

print('Énergie consommée après optim : %.2f kJ soit %.1f %% E0' % (E_best/1000, E_best/E0*100))

# Tracé du profil de vitesse :
plt.figure('optim vitesse')
plt.plot(l,v_best)
plt.hlines(v_min,0,L, colors='blue', linestyles='dashed')
plt.title(u'Vitesse : profil optimisé (%d itérations)' % n_iter)
plt.xlabel('abscisse l [m]')
plt.ylabel('vitesse [m/s]')
plt.grid(True)
plt.twinx()
plt.plot(l,z, 'g')
plt.xlabel('altitude z [m]')

# Historique des optimum partiels
plt.figure('optim hist')
plt.subplot(211)
plt.hist(np.array(E_list)/E0*100, normed=True)
plt.title(u'Optimum intermédiaires (%d itérations)' % n_iter)
plt.vlines(100, *plt.ylim(), colors='r')
plt.grid(True)
plt.subplot(212)
plt.plot(i_list, E_list/E0*100)
plt.ylabel('E_best/E0 [%]')
plt.xlim((0, n_iter))
plt.hlines(100, 0, n_iter, colors='r')
plt.grid(True)

plt.show()
