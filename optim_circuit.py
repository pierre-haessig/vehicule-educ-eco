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

# Profil de vitesse initial :
v0 = np.ones(N)*v_min # vitesse constante

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
    sera inchangée
    Méthode de génération : bruit AR(1)
    '''
    N = len(v_prev)
    scale = np.random.rayleigh(0.004)
    corr = 1-10**np.random.uniform(-8,-2)
    corr = 1-10**-3
    # corr = 0.99999
    #scale = 0.1
    noise = np.random.normal(size=N, scale=scale)
    # Filtrage passe-bas, normalisé en puissance
    noise = lfilter([np.sqrt(1-corr**2)],[1. , -corr], noise)
    v = v_prev + (noise - noise.mean())
    return v, scale, corr

def random_v2(v_prev):
    '''génère un vecteur vitesse aléatoirement
    en se basant sur un vecteur vitesse dont la moyenne
    sera inchangée
    Méthode de génération : coup d'accélérateur aléatoire
    '''
    N = len(v_prev)
    scale = np.random.rayleigh(0.0005)
    scale = np.random.normal(scale=0.001)
    
    # Tirage de la largeur du "coup de boost"
    # entre 0 et 1, avec un exposant qui augmente l'occurence des faibles valeurs
    largeur = (np.random.uniform()**2)
    demi_largeur = int(largeur*N/2)+1
    
    accel = np.zeros(N)    
    # Génération du "coup de boost"
    boost = (np.cos(np.arange(-demi_largeur,demi_largeur+1)*np.pi/demi_largeur)+1)*(scale/2)
    # Tirage aléatoire de la localisation
    x0 = np.random.randint(N)
    x_boost = np.arange(x0-demi_largeur, x0+demi_largeur+1) % N
    accel[x_boost] = boost
    
    # Recentrer l'accélération:
    accel -= accel.mean()
    
    # Intégration discrète:
    v = accel.cumsum()
    # Recentrer la vitesse :
    v -= v.mean()
    
    return v + v_prev, scale, largeur

def random_search(niter):
    '''recherche aléatoire d'un minimiseur'''
    np.random.seed(0)
    v0 = np.ones(N)*v_min
    E = crit_E_tot(v0)
    v_best = v0
    E_best = E
    # Mémoire des améliorations successives :
    E_list = []
    i_list = []
    scale_list = []
    corr_list = []
    for i in xrange(niter):
        v,scale,corr = random_v2(v_best)
        E = crit_E_tot(v)
        
        if E < E_best:
            # Sauvegarder le progrès
            E_best = E
            v_best = v
            E_list.append(crit_E_tot(v, with_penalty=False))
            i_list.append(i)
            scale_list.append(scale)
            corr_list.append(corr)
    
    return (v_best, E_best, E_list, i_list, scale_list, corr_list)

# Lancement de l'optimisation
if __name__=='__main__':
    n_iter = 10**4
    print('Optimisation aléatoire en %d itérations...' %n_iter)
    result = random_search(n_iter)
    v_best, E_best, E_list, i_list, scale_list, corr_list = result
    E_best = crit_E_tot(v_best, with_penalty=False)
    print('itérations apportant une amélioration : %d (%.1f %%)' % 
          (len(E_list), len(E_list)/n_iter*100 ))

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
