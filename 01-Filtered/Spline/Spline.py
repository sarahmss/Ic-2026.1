import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

TITLES = ["Train_1", "Train_2", "Test_1", "Test_2", "Test_3", "Val", "LSG_1", "LSG_2"]
TS = 0.07
os.chdir("./01-Filtered")
#(k, s)
parametros_spline = {
    "WdRef": (3, 0.125),
    "WeRef": (3, 0.125),
    "Wd": (3, 0.125),
    "We": (3, 0.125),
    "PwmD": (3, 0.025),
    "PwmE": (3, 0.025),
    "Theta": (3, 0.025),
    "X": (3, 0.01),
    "Y": (3, 0.01),
}

# Writer para salvar múltiplas sheets
output_path = "./Data/SplinesDatasets.xlsx"
writer = pd.ExcelWriter(output_path, engine="openpyxl")

for title in TITLES:
    print(f"Processando: {title}")
    
    # Lê dataset
    df = pd.read_excel("./Data/Datasets.xlsx", sheet_name=title)
    
    # Assume que a primeira coluna é tempo
    tempo = np.arange(len(df)) * TS 
    
    data_spline = pd.DataFrame()
    data_spline[df.columns[0]] = tempo  # mantém coluna de tempo
    
    splines = {}

    # Aplica spline nas colunas especificadas
    for col, (k, fator_s) in parametros_spline.items():
        if col not in df.columns:
            continue

        y = df[col].to_numpy()
        s_val = len(tempo) * np.var(y) * fator_s

        spline = UnivariateSpline(tempo, y, s=s_val, k=k)
        splines[col] = spline

        data_spline[col] = spline(tempo)

        # (Opcional) Plot
        tempo_fino = np.linspace(tempo.min(), tempo.max(), 1000)
        y_interp = spline(tempo_fino)

        # plt.figure(figsize=(8, 3))
        # plt.plot(tempo, y, 'o', label='Real', alpha=0.5)
        # plt.plot(tempo_fino, y_interp, '-', label=f"Spline k={k}, s={fator_s}")
        # plt.title(f"{title} - {col}")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

    # Salva sheet
    data_spline.to_excel(writer, sheet_name=title, index=False)

# Salva arquivo final
writer.close()

print(f"Arquivo salvo em: {output_path}")