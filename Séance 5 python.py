#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np



#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))

# Renommer les colonnes si nécessaire
donnees.columns = ["Pour", "Contre", "Sans opinion"]

# Moyenne des effectifs sur les 100 échantillons
moyennes = donnees.mean().round()

print("\nMoyennes observées sur 100 échantillons :")
print(moyennes)

# Fréquences observées
total_moy = moyennes.sum()
freq_obs = (moyennes / total_moy).round(2)

print("\nFréquences observées :")
print(freq_obs)

# Fréquences de la population mère
pop_mere = {"Pour": 852, "Contre": 911, "Sans opinion": 422}
total_pop = sum(pop_mere.values())
freq_pop = {k: round(v/total_pop, 2) for k, v in pop_mere.items()}

print("\nFréquences de la population mère :")
print(freq_pop)

# Calcul de la taille moyenne des échantillons
n_moy = donnees.sum(axis=1).mean().round()
n_moy = int(n_moy)

print(f"\nTaille d'échantillon moyenne : n = {n_moy}")

# Intervalle de fluctuation à 95%
z = 1.96

print("\nIntervalles de fluctuation (95%) :")
for modalite in moyennes.index:
    p = freq_obs[modalite]
    se = math.sqrt(p*(1-p)/n_moy)
    borne_inf = max(0, p - z*se)
    borne_sup = min(1, p + z*se)
    print(f"{modalite} : [{borne_inf:.4f} ; {borne_sup:.4f}]")


#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

# On prend le premier échantillon (ligne 0)
echantillon = donnees.iloc[0]
n = int(echantillon.sum())

print("\nPremier échantillon :")
print(echantillon)

print(f"\nTaille de l'échantillon : n = {n}")

# Fréquences dans l’échantillon
freq_ech = echantillon / n
print("\nFréquences dans l'échantillon :")
print(freq_ech)

# IC95%
print("\nIntervalles de confiance (95%) :")
for modalite in echantillon.index:
    p = freq_ech[modalite]
    se = math.sqrt(p*(1-p)/n)
    borne_inf = max(0, p - z*se)
    borne_sup = min(1, p + z*se)
    print(f"{modalite} : [{borne_inf:.4f} ; {borne_sup:.4f}]")


#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

# Chargement des deux séries
serie1 = ouvrirUnFichier("./data/Loi-normale-Test-1.csv")
serie2 = ouvrirUnFichier("./data/Loi-normale-Test-2.csv")

# Conversion en vecteurs
s1 = serie1.values.flatten()
s2 = serie2.values.flatten()

print("\nTaille des séries :")
print("Série 1 :", len(s1))
print("Série 2 :", len(s2))

# Test de Shapiro-Wilk
W1, p1 = scipy.stats.shapiro(s1)
W2, p2 = scipy.stats.shapiro(s2)

print("\nTest de Shapiro-Wilk :")
print(f"Série 1 : W={W1:.4f}, p={p1:.4f}")
print(f"Série 2 : W={W2:.4f}, p={p2:.4f}")

alpha = 0.05

if p1 > alpha:
    print(" → La série 1 peut être considérée comme normale.")
else:
    print(" → La série 1 n'est PAS normale.")

if p2 > alpha:
    print(" → La série 2 peut être considérée comme normale.")
else:
    print(" → La série 2 n'est PAS normale.")

# Bonus : proposer une autre distribution si non normale
print("\nBonus : proposition de distribution pour celles non normales")

def teste_loi(data, loi):
    dist = getattr(scipy.stats, loi)
    params = dist.fit(data)
    D, p = scipy.stats.kstest(data, loi, args=params)
    return p, params

lois_candidates = ["expon", "lognorm", "gamma", "weibull_min"]

for label, serie, p in [("Série 1", s1, p1), ("Série 2", s2, p2)]:
    if p > alpha:
        print(f"{label} : déjà normale → pas de bonus.")
    else:
        print(f"\n{label} non normale → test d'ajustement sur lois candidates :")
        meilleurs = []
        for loi in lois_candidates:
            try:
                p_val, params = teste_loi(serie, loi)
                meilleurs.append((p_val, loi, params))
                print(f"{loi} : p={p_val:.4f}, params={params}")
            except:
                print(f"{loi} : impossible d'ajuster")
        meilleurs.sort(reverse=True)
        print(f" → Meilleure loi candidate : {meilleurs[0][1]} (p={meilleurs[0][0]:.4f})")


# ============================
# GRAPHIQUES CÔTE À CÔTE
# ============================

# Pour dessiner deux graphiques dans la même fenêtre
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Données
x1 = s1
x2 = s2

# ============================
# Graphique 1 : Série 1
# ============================
axes[0].hist(x1, bins=20, density=True, alpha=0.6, color="skyblue", label="Histogramme")
mu1, sigma1 = np.mean(x1), np.std(x1)
xx1 = np.linspace(x1.min(), x1.max(), 300)
axes[0].plot(xx1, scipy.stats.norm.pdf(xx1, mu1, sigma1), linewidth=2, label="Densité normale")

axes[0].set_title("Distribution - Série 1")
axes[0].set_xlabel("Valeurs")
axes[0].set_ylabel("Densité")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# ============================
# Graphique 2 : Série 2
# ============================
axes[1].hist(x2, bins=20, density=True, alpha=0.6, color="lightgreen", label="Histogramme")
mu2, sigma2 = np.mean(x2), np.std(x2)
xx2 = np.linspace(x2.min(), x2.max(), 300)
axes[1].plot(xx2, scipy.stats.norm.pdf(xx2, mu2, sigma2), linewidth=2, label="Densité normale")

axes[1].set_title("Distribution - Série 2")
axes[1].set_xlabel("Valeurs")
axes[1].set_ylabel("Densité")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

# Affichage
plt.suptitle("Comparaison visuelle des deux distributions", fontsize=16)
plt.tight_layout()
plt.show()
