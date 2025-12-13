#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("data/resultats-elections-presidentielles-2022-1er-tour.csv", encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)

# Mettre dans un commentaire le numéro de la question
# Question 1
#...
print(pd.DataFrame(contenu))
df=pd.DataFrame(contenu)
print(len(contenu))
print(contenu.head(0))


nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
print("Nombre de ligne :", nb_lignes)
print("Nombre de colonnes :", nb_colonnes)
types_colonnes = []

for col in contenu.columns:
    dtype = contenu[col].dtype
    if pd.api.types.is_integer_dtype(dtype):
        types_colonnes.append('int')
    elif pd.api.types.is_float_dtype(dtype):
        types_colonnes.append('float')
    elif pd.api.types.is_bool_dtype(dtype):
        types_colonnes.append('bool')
    else:
        types_colonnes.append('str')

print(contenu.dtypes)
print(contenu.columns)
print(contenu.head())

colonne_inscrits = contenu["Inscrits"]
print(colonne_inscrits)

# Étape 10 — Calculer la somme des colonnes quantitatives

print("\n=== Somme des colonnes quantitatives ===")
somme_colonnes = {}

for col, typ in zip(contenu.columns, types_colonnes):
    if typ in ('int', 'float'):
        somme = contenu[col].sum()
        somme_colonnes[col] = somme

print(somme_colonnes)

# Etape 11

for i in range(len(contenu)): #=>en haut = la boucle
    dept= contenu.loc[i, "Libellé du département"]
    inscrits = contenu.loc[i, "Inscrits"]
    votants = contenu.loc[i, "Votants"]
    plt.figure(figsize=(6,4)) #le diagramme
    plt.bar(["Inscrits", "Votants"], [inscrits, votants], color=['green', 'blue'])
    plt.title(f"{dept}")
    plt.ylabel("Nombre de personnes")
    plt.ticklabel_format(style='plain', axis='y')
    #avoir les noms et par les valeurs de matplotilib
    plt.savefig(f"{dept}.png")
    plt.close()

# Étape 12 : Diagrammes circulaires des votes par département
dossier_images = "images_diagrammes"
import os 

os.makedirs("images_circulaires", exist_ok=True)

for i in range(len(contenu)): #la boucle pour chaque département
    dep = contenu.loc[i, "Libellé du département"]
    blancs = contenu.loc[i, "Blancs"]
    nuls = contenu.loc[i, "Nuls"]
    votants = contenu.loc[i, "Votants"]
    abstention = contenu.loc[i, "Abstentions"]

    exprimés = votants - blancs - nuls

 

    valeurs = [blancs, nuls, exprimés, abstention]
    labels = ["Blans", "Nuls", "Exprimés", "Abstentions"]
    couleurs = ["grey", "red", "green", "blue"]

    plt.figure(figsize=(6,6))
    plt.pie(valeurs, labels=labels, autopct='%1.1f%%', startangle=90, colors=couleurs)
    plt.title(f"Répartition des votes - {dep}")
    plt.tight_layout()
    plt.savefig(f"images_circulaires/{dep}.png")
    plt.close()

# Etape 13

import os
import matplotlib.pyplot as plt

def nettoyer_nom(nom):
    nom = nom.replace("/", "_")
    nom = nom.replace(" ", "_")
    nom = nom.replace("'", "_")
    return nom

for dept in contenu['Libellé du département'].unique():
    dept_clean = nettoyer_nom(dept)
    dossier_images = f"histogrammes/{dept_clean}"
    os.makedirs(dossier_images, exist_ok=True)

    inscrits_dept = contenu[contenu['Libellé du département'] == dept]['Inscrits']

    plt.figure(figsize=(8,5))
    plt.hist(inscrits_dept)
    plt.title(f"Histogramme des inscrits - {dept}")
    plt.xlabel("Nombre d'inscrits")
    plt.ylabel("Fréquence")

    plt.savefig(f"{dossier_images}/{dept_clean}.png")
    plt.close()

print("Histogrammes créés pour tous les départements.")
