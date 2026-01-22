import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
import seaborn as sns

# 1. importation et exploration
df = pd.read_csv('data_insurance.csv')
print("structure des donnees :")
print(df.info())
print("\n", df.describe())
print("\nvaleurs manquantes :", df.isnull().sum().sum())




# 2. étude a : bmi vs région (anova)

print("étude a : bmi depend-il de la region ?")

# question 1 : boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='region', y='bmi', palette='Set2')
plt.title('bmi par region')
plt.ylabel('bmi')
plt.xlabel('region')
plt.tight_layout()
plt.savefig('boxplot_bmi_region.png')
plt.show()
# ---
# analyse visuelle

print('''\nobservation : les boîtes ont des hauteurs grossièrement similaires,
      indiquant une variabilité comparable du bmi entre les régions.''')

# question 2 : statistiques descriptives
print("\nmoyennes et ecarts-types du bmi par region :")
stats_bmi = df.groupby('region')['bmi'].agg(['mean', 'std', 'count'])
print(stats_bmi.round(2))
#--
if (stats_bmi['std'].max() / stats_bmi['std'].min() < 1.2):
    print("les ecarts-types sont tres proches, on peut donc utiliser l'anova")
else:
    print("les ecarts-types sont pas tres proches, on ne peut pas utiliser l'anova")
#--

# question 3 : test anova
groupes = [groupe['bmi'].values for region ,  groupe in df.groupby('region')]
f_stat, p_value = f_oneway(*groupes)
print(f"\nanova : f-statistique = {f_stat:.4f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("rejet de h0 : au moins une region a un bmi moyen significativement different")
else:
    print("acceptation de h0 : les bmi moyens sont identiques entre regions")

# question 4 : région à risque
region_risque = stats_bmi['mean'].idxmax()
print(f"\nregion la plus a risque (bmi eleve) : {region_risque} (bmi = {stats_bmi.loc[region_risque, 'mean']:.2f})")

# 3. étude b : tabagisme vs région (chi-2)
print("étude b : le tabagisme depend-il de la region ?")

# question 1 : tableau de contingence
contingence = pd.crosstab(df['region'], df['smoker'])
print("\ntableau de contingence (fumeurs/non-fumeurs par region) :")
print(contingence)


# question 2 : test du chi-2
chi2, p_value_chi2, dof, freq_attendues = chi2_contingency(contingence)
print(f"\nchi-2 : x² = {chi2:.4f}, p-value = {p_value_chi2:.6f}")
if p_value_chi2 < 0.05:
# question 3 : proportions par région
    print("rejet de h0 : il y a une dependance entre region et tabagisme")
else:
    print("acceptation de h0 : les variables sont independantes")

print("\nproportion de fumeurs par region :")
prop_fumeurs = df.groupby('region')['smoker'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).round(2)
print(prop_fumeurs)

print(stats_bmi)

# 4. synthèse managériale
print("synthese pour la direction")

conclusion_bmi = "OUI" if p_value < 0.05 else "NON"
conclusion_tabac = "OUI" if p_value_chi2 < 0.05 else "NON"

print(f"""
1. impact sur l'obesite (bmi) : {conclusion_bmi}
    lieu d'habitation {'influence' if conclusion_bmi == 'OUI' else "n'influence pas"} 
    significativement l'imc (p={p_value:.4f})

2. impact sur le tabagisme : {conclusion_tabac}
   - le lieu d'habitation {'influence' if conclusion_tabac == 'OUI' else "n'influence pas"} 
     significativement le tabagisme (p={p_value_chi2:.4f})

3. recommandation : 
   moduler les prix par region {'oui, car au moins un facteur varie' if (p_value < 0.05 or p_value_chi2 < 0.05) else 'non, facteurs homogenes'}
   region a surveiller : {region_risque} (bmi moyen = {stats_bmi.loc[region_risque, 'mean']:.2f})
""")

