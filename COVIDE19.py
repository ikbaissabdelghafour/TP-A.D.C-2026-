import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

covide_data = pd.read_excel('./covide_dataset.xlsx')
df = covide_data.copy()  # Create a copy of the dataset
def EX1():
    print('!  Exercice 1 : Initialisation et Analyse de Forme   ')
    print(df.head(5))
    print(df.shape)
    print(df.dtypes.value_counts())
def EX2():
    print('!  Exercice 2 : Gestion des valeurs manquantes (NaN)  ')
    
    # 1. Heatmap des valeurs manquantes
    print('\n1. Heatmap des valeurs manquantes:')
    plt.figure(figsize=(18, 8))
    sns.heatmap(df.isna(), cbar=True, cmap='viridis')
    plt.title('Heatmap des valeurs manquantes')
    plt.savefig('Covide_Figures/EX2_Heatmap des valeurs manquantes')
    plt.show()
    # 2. Pourcentage des valeurs manquantes
    print('Pourcentage des valeurs manquantes.')
    print((df.isna().sum()/len(df))*100)
    # 3. Suppression des colonnes avec plus de 90% de valeurs manquantes
    print('Suppression des colonnes avec plus de 90% de valeurs manquantes')
    df.dropna(axis=1 , thresh=len(df)*0.1 ,inplace=True)
    print((df.isna().sum()/len(df))*100)
    print(df.shape)
def EX3():
    #? ----------- 1 --------------
    print('!  Exercice 3 : Visualisation des données  ')
    distrubution_SARS =(df['SARS-Cov-2 exam result'].value_counts(normalize=True)*100).round()
    print(distrubution_SARS)
    #? ----------- 2 --------------
    print('''On privilégie le F1-score car il représente un bon compromis entre
    la précision et le rappel, notamment dans le cas d'un dataset déséquilibré.''')
def Ex4():
    print('!  Exercice 4 : Analyse des variables (Histogrammes)   ')
    # 1. Isoler colonnes float et object
    sang_var = df.select_dtypes(include='float64')
    viral_vars = df.select_dtypes(include='object')
    print(f"\nVariables sanguines: {list(sang_var)}")
    print(f"Variables virales: {list(viral_vars)}")
    # 2. Distribution de chaque variable sanguine
    print("\n Distribution des variables sanguines ")
    for col in sang_var:
        sns.histplot(data=sang_var, x=col, kde=True, bins=30)
        plt.title(f'{col}\n Moyenne : {df[col].mean():.2f} , l\'ecarte-type: {df[col].std():.2f}')
        plt.savefig(f'Covide_Figures/EX4_{col} Distribution des variables sanguines ')
        plt.show()
    # Interpretation
    print('''Les données ne sont pas standardisées car les distributions ne sont pas centrées autour de 0 
          et les échelles des variables sont différentes.''')
    # 3. Relation avec Target (positif/négatif)
    print("\n Relation avec résultat COVID ")
    target = 'SARS-Cov-2 exam result'
    for col in sang_var:
        plt.figure(figsize=(10, 5))
        for result in df[target].unique():
            if pd.notna(result):
                sns.histplot(data=df[df[target] == result], x=col, kde=True, label=result, alpha=0.6)
        plt.title(f'{col} selon résultat COVID')
        plt.legend()
        plt.savefig(f'Covide_Figures/EX4_{col} selon résultat COVID')
        plt.show()
    # Interpretation
    print('''Les variables qui montrent une séparation entre les cas positifs et négatifs sont :
          principalement les leucocytes, les plaquettes, les lymphocytes, les monocytes et les éosinophiles.''')
        
        
def Ex5():
    print(' Exercice 5 : Variables Catégorielles  ')
    target = 'SARS-Cov-2 exam result'
    viral_cols = [col for col in df.select_dtypes(include='object').columns if col != target]
    # 1. pd.crosstab pour relation Target vs virus
    print("\n1. Relation Target vs Virus (crosstab):")
    for virus in viral_cols[:2]:  # 2 premiers virus
        ct = pd.crosstab(df[target], df[virus])
        print(f"\n{virus}:\n{ct}") 
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd')
        plt.title(f'COVID vs {virus}')
        plt.savefig(f'Covide_Figures/EX5_COVID vs {virus}')
        plt.show()
    
    # 2. Co-infections
    print("\n2. Co-infections (plusieurs virus):")
    df['nb_virus'] = (df[viral_cols] == 'positive').sum(axis=1)
    print(f"Patients avec >1 virus: {(df['nb_virus'] > 1).sum()} ({(df['nb_virus'] > 1).sum()/len(df)*100:.1f}%)")
    
    # 3. Nouvelle colonne est_malade
    print("\n3. Colonne 'est_malade' vs COVID:")
    df['est_malade'] = (df[viral_cols] == 'positive').any(axis=1)
    comparison = pd.crosstab(df[target], df['est_malade'])
    print(comparison)
    sns.heatmap(comparison, annot=True, fmt='d', cmap='Blues')
    plt.title('COVID vs Autre virus')
    plt.savefig('EX5_COVID vs Autre virus')
    plt.show()

def Ex6():
    print('Exercice 6 : Corrélations et Tests Statistiques  ')
    target = 'SARS-Cov-2 exam result'
    sang_var = df.select_dtypes(include='float64').columns
    
    # 1. Matrice de corrélation
    print("\n1. Matrice de corrélation des variables sanguines:")
    corr_matrix = df[sang_var].corr()
    print(corr_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.savefig('Covide_Figures/EX6_corrélation des variables sanguines')
    plt.title('Matrice de Corrélation')
    plt.show()
    # - interpretation 
    print('''Après observation de la matrice de corrélation, les variables les plus corrélées
          correspondent aux paires situées sur la diagonale principale de la matrice, 
          qui présentent un coefficient de corrélation égal à r = 1,
          ce qui indique une corrélation parfaite de chaque variable avec elle-même.''')
    
    # 2 & 3. Test de Student (t-test) positif vs négatif
    print("\n2-3. Test de Student (p-value < 0.05 = significatif):")
    positive = df[df[target] == 'positive']
    negative = df[df[target] == 'negative']
    
    for col in sang_var:
        t_stat, p_value = ttest_ind(positive[col].dropna(), negative[col].dropna())
        sig = "SIGNIFICATIF" if p_value < 0.05 else "No"
        print(f"{col}: p-value = {p_value:.4f} {sig}")
    print('''Une p-value < 0,05 confirme une différence statistiquement significative entre groupes,
          cohérente avec les distributions décalées observées à l’exercice 4''')
    
EX1()
EX2()
EX3()
Ex4()
Ex5()
Ex6()
