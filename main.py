import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_missing.iloc[:, 1:]),
    columns=df_missing.columns[1:]
)

df_imputed = pca_imputation(df_scaled, max_iteration = 10, n_components = 3, tolerance = 0.001)

df_imputed_original_scale = df_missing.copy()
df_imputed_original_scale.iloc[:, 1:] = scaler.inverse_transform(df_imputed)
mask_missing = df_missing.isna()

imputed_values = df_imputed_original_scale.iloc[:, 1:].values[mask_missing.iloc[:, 1:].values]
original_values = df_full.iloc[:, 1:].values[mask_missing.iloc[:, 1:].values]

mse = mean_squared_error(original_values, imputed_values)
print(f"Mean Squared Error (MSE) medzi originálnymi a doplnenými hodnotami: {mse}")

mean_imputed = df_missing.fillna(df_missing.mean())## Imputácia priemerom

mask_missing_no_id = mask_missing.iloc[:, 1:]

original_values_mean = df_full.iloc[:, 1:].values[mask_missing_no_id.values]
imputed_values_mean = mean_imputed.iloc[:, 1:].values[mask_missing_no_id.values]

## MSE pre imputáciu priemerom
mse_mean = mean_squared_error(original_values_mean, imputed_values_mean)
print(f"MSE pri imputácii priemerom: {mse_mean}")


#pca = PCA()
#pca.fit(df_full)
#explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
#print(f"Kumulatívna variancia: {explained_variance_ratio}")
