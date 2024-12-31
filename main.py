import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

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

df_inputed = pca_imputation(df_missing, max_iteration = 10,n_components = 1, tolerance = 0.001)

mask_missing = df_missing.isna()
imputed_values = df_inputed[mask_missing].values
original_values = df_full[mask_missing].values

#mse = mean_squared_error(original_values, inputed_values)
#print(f"Mean Squared Error (MSE) medzi originálnymi a doplnenými hodnotami: {mse}")
# Overíme, či nie sú NaN v pôvodných a imputovaných hodnotách
print("Sú NaN v original_values?", np.isnan(original_values).any())
print("Sú NaN v imputed_values?", np.isnan(imputed_values).any())
#pca = PCA()
#pca.fit(df_full)
#explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
#print(f"Kumulatívna variancia: {explained_variance_ratio}")
