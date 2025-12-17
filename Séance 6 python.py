#coding:utf8

import csv
import math
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


def ouvrirUnFichier(chemin):
    with open(chemin, newline='', encoding='utf-8') as f:
        return list(csv.reader(f, delimiter=','))


def ordreDecroissant(liste):
    return sorted(liste, reverse=True)


def conversionLog(liste):
    return [math.log10(x) for x in liste]


def ordrePopulation(valeurs, etats):
    couples = list(zip(valeurs, etats))
    couples.sort(reverse=True)
    return [(etat, rang + 1) for rang, (_, etat) in enumerate(couples)]


def classementPays(classement_pop, classement_dens):
    d_pop = {etat: rang for etat, rang in classement_pop}
    d_dens = {etat: rang for etat, rang in classement_dens}
    resultat = [(etat, d_pop[etat], d_dens[etat]) for etat in d_pop if etat in d_dens]
    resultat.sort(key=lambda x: x[1])
    return resultat


iles = ouvrirUnFichier('data\island-index.csv')
print(iles)
header_iles = iles[0]
data_iles = iles[1:]
idx_surface = [i for i, h in enumerate(header_iles) if 'Surface' in h][0]
surfaces = [float(ligne[idx_surface]) for ligne in data_iles]
surfaces += [85545323.0, 37856841.0, 7768030.0, 7605049.0]
surfaces_ord = ordreDecroissant(surfaces)
rangs = list(range(1, len(surfaces_ord) + 1))

plt.figure()
plt.plot(rangs, surfaces_ord)
plt.savefig('rang_taille.png')
plt.close()

surfaces_log = conversionLog(surfaces_ord)
rangs_log = conversionLog(rangs)

plt.figure()
plt.plot(rangs_log, surfaces_log)
plt.savefig('rang_taille_log.png')
plt.close()

#Il n'est pas possible de faire un test sur les rangs car il n'y a qu'une seule varaible ordonnée. 

#7

monde = ouvrirUnFichier('data/Le-Monde-HS-Etats-du-monde-2007-2025.csv')
header = monde[0]
data = monde[1:]

idx_etat = header.index('État')
idx_pop2007 = header.index('Pop 2007')
idx_pop2025 = header.index('Pop 2025')
idx_den2007 = header.index('Densité 2007')
idx_den2025 = header.index('Densité 2025')

etats = []
pop2007 = []
pop2025 = []
den2007 = []
den2025 = []


for l in data:
    if l[idx_pop2007] != '' and l[idx_den2007] != '' and l[idx_pop2025] != '' and l[idx_den2025] != '':
        etats.append(l[idx_etat])
        pop2007.append(float(l[idx_pop2007]))
        pop2025.append(float(l[idx_pop2025]))
        den2007.append(float(l[idx_den2007]))
        den2025.append(float(l[idx_den2025]))

classe_pop2007 = ordrePopulation(pop2007, etats)
classe_pop2025 = ordrePopulation(pop2025, etats)
classe_den2007 = ordrePopulation(den2007, etats)
classe_den2025 = ordrePopulation(den2025, etats)

comp2007 = classementPays(classe_pop2007, classe_den2007)
comp2025 = classementPays(classe_pop2025, classe_den2025)

r_pop2007 = [x[1] for x in comp2007]
r_den2007 = [x[2] for x in comp2007]

r_pop2025 = [x[1] for x in comp2025]
r_den2025 = [x[2] for x in comp2025]

spearman_2007 = spearmanr(r_pop2007, r_den2007)
kendall_2007 = kendalltau(r_pop2007, r_den2007)

spearman_2025 = spearmanr(r_pop2025, r_den2025)
kendall_2025 = kendalltau(r_pop2025, r_den2025)
