import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Funkcia na imputáciu pomocou PCA
def pca_imputation(df, n_components=5, max_iter=10, tol=1e-4):
    df_filled = df.copy()
    df_filled[:] = df_filled.fillna(df.mean())  # Inicializácia imputácie jednoduchou metódou

    for iteration in range(max_iter):
        # PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_filled)
        df_reconstructed = pd.DataFrame(
            pca.inverse_transform(principal_components),
            index=df_filled.index,
            columns=df_filled.columns
        )

        # Aktualizácia len tam, kde sú pôvodne chýbajúce hodnoty
        prev_filled = df_filled.copy()
        mask = df.isna()  # Maskovanie pôvodne chýbajúcich hodnôt
        df_filled[mask] = df_reconstructed[mask]

        # Kontrola konvergencie
        diff = np.linalg.norm(df_filled.values - prev_filled.values)
        print(f"Iterácia {iteration + 1}, rozdiel: {diff:.6f}")
        if diff < tol:
            break

    return df_filled


# Príklad použitia
# Načítanie dát (simulácia)
df_missing = pd.read_csv("v5/v5_missing.csv")
df_full = pd.read_csv("v5/v5_complete.csv")

# Normalizácia (iba numerické dáta)
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_missing.iloc[:, 1:]),
    columns=df_missing.columns[1:]
)

# PCA imputácia
df_imputed = pca_imputation(df_scaled)

# Opačná transformácia na pôvodnú mierku
df_imputed_original_scale = pd.DataFrame(
    scaler.inverse_transform(df_imputed),
    columns=df_missing.columns[1:]
)

# Porovnanie s originálnymi dátami (ak máme plné dáta)
mse = mean_squared_error(
    df_full.iloc[:, 1:].dropna().values,
    df_imputed_original_scale.dropna().values
)
print(f"MSE medzi originálnymi a imputovanými dátami: {mse}")

plt.scatter(df_full['x_3'], df_imputed_original_scale['x_3'], alpha=0.5)
plt.xlabel('Originálne hodnoty')
plt.ylabel('Imputované hodnoty')
plt.title('Porovnanie originálnych a imputovaných hodnôt pre x_3')
plt.show()