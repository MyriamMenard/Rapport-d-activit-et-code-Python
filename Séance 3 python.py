#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Séance 3

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023
df = pd.read_csv("data/resultats-elections-presidentielles-2022-1er-tour.csv", encoding="utf-8")
print(df.head())

# Etape 5 : 

quant_cols = ["Inscrits","Abstentions","Votants","Blancs","Nuls","Exprimés","Voix"]
quant_df = df[quant_cols]

results = {}
for col in quant_cols:
    series = quant_df[col].dropna()
    results[col] = {
        "Moyenne": round(series.mean(), 2),
        "Médiane": round(series.median(), 2),
        "Mode": round(series.mode().iloc[0], 2) if not series.mode().empty else np.nan,
        "Écart type": round(series.std(), 2),
        "Écart absolu à la moyenne": round((abs(series - series.mean())).mean(), 2),
        "Étendue": round(series.max() - series.min(), 2),
        "Distance interquartile": round(series.quantile(0.75) - series.quantile(0.25), 2),
        "Distance interdécile": round(series.quantile(0.9) - series.quantile(0.1), 2)
    }

# Etape 6
for col, stats in results.items():
    print(f"\nParamètres pour {col}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

# Etape 7 

plt.figure(figsize=(12,8))
quant_df.boxplot()
plt.title("Boîtes de dispersion des colonnes quantitatives")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boites_dispersion.png")  # Sauvegarde l'image
plt.show()

# Etape 8 

results_df = pd.DataFrame(results).T
results_df.to_csv("parametres_elections.csv", index=True)
results_df.to_excel("parametres_elections.xlsx", index=True)


# Etape 9

with open("data/island-index.csv", encoding="utf-8") as f:
    contenu = pd.read_csv(f)

print(pd.DataFrame(contenu))
df=pd.DataFrame(contenu)



# Étape 9 : Catégoriser les îles par surface
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, float("inf")]
labels = [
    "0-10 km2",
    "10-25 km2",
    "25-50 km2",
    "50-100 km2",
    "100-2500 km2",
    "2500-5000 km2",
    "5000-10000 km2",
    ">=10000 km2"
]

df["Categorie"] = pd.cut(df["Surface (km²)"], bins=bins, labels=labels, right=True)

# Étape 10 : Compter le nombre d’îles par catégorie
counts = df["Categorie"].value_counts().sort_index()

print("\nNombre d'îles par catégorie de surface :")
print(counts)

# Bonus : Visualiser avec un graphique en barres
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Répartition des îles par catégorie de surface")
plt.xlabel("Catégorie de surface")
plt.ylabel("Nombre d'îles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
