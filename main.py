import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df_missing = pd.read_csv("v5/v5_missing.csv")
df_full = pd.read_csv("v5/v5_complete.csv")

def pca_imputation(df, max_iteration = 10,n_components = 5, tolerance = 0.001):
    copy_of_df = df.copy() ##urobím kópiu vstupneho dataframeu
    copy_of_df[:] = copy_of_df.fillna(df.mean()) ##vyplnim chybajuce data priemerom

    for i in range(max_iteration):
        pca = PCA(n_components = n_components)
        principal_components = pca.fit_transform(copy_of_df) ##vykoná sa tranformácia dát do priestoru komponentov
        df_reconstructed = pd.DataFrame(
            pca.inverse_transform(principal_components),
            index = copy_of_df.index,
            columns = copy_of_df.columns
        ) ## nasledne sa rekonštruuju dáta na pôvodný tvar dataframeu
        previev_filled = copy_of_df.copy()
        mask = df.isna() ## vytvára maticu, ktorá ma miesta kde boli nezadané hodnoty NA
        copy_of_df[mask] = df_reconstructed[mask] ## tam kde chybali hodnoty ich nahradíme rekonštukturovanými hodnotamy
        difference = np.linalg.norm(copy_of_df.values - previev_filled.values) ## ak sa velkost zmeny neposunula ako je nastavená tolerancia tak končime
        if difference < tolerance:
            break
    return copy_of_df

pca = PCA()
pca.fit(df_full)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
print(f"Kumulatívna variancia: {explained_variance_ratio}")
