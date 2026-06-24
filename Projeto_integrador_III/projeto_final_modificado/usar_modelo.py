# =============================================================
# PROJETO: PredictInd - Monitoramento Preditivo de Compressor
# ARQUIVO: usar_modelo.py
# DESCRIÇÃO: Usa o modelo treinado para inferir dados e gerar plots
# DISCIPLINA: Projeto Integrador Big Data na Indústria III
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

ARQUIVO_DADOS    = "dados_uso.csv"
ARQUIVO_MODELO   = "modelo_manutencao.pkl"
ARQUIVO_SCALER   = "scaler_manutencao.pkl"
ARQUIVO_SAIDA    = "resultado_uso_ia.csv"

DESCRICAO_CLASSE = {0: "Operação Normal", 1: "Atenção", 2: "⚠ RISCO DE FALHA ⚠"}
ICONE_CLASSE = {0: "🟢", 1: "🟡", 2: "🔴"}

print("=" * 65)
print("  PredictInd - Uso do Modelo de IA em Novos Dados")
print("=" * 65)

modelo = joblib.load(ARQUIVO_MODELO)
scaler = joblib.load(ARQUIVO_SCALER)
df_uso = pd.read_csv(ARQUIVO_DADOS)

X_uso_norm = scaler.transform(df_uso)
predicoes_codigo = modelo.predict(X_uso_norm)
probabilidades = modelo.predict_proba(X_uso_norm)

df_resultado = df_uso.copy()
df_resultado["predicao_codigo"] = predicoes_codigo
df_resultado["status_previsto"] = [DESCRICAO_CLASSE[c] for c in predicoes_codigo]
df_resultado["confianca_pct"] = [f"{max(prob)*100:.1f}%" for prob in probabilidades]

contagem = pd.Series(predicoes_codigo).value_counts().sort_index()

print("\n[PASSO 9] Gerando gráfico dos resultados...")

# ----- GRÁFICO 1: Pizza -----
labels_presentes = []
valores_presentes = []
cores_mapa = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
cores_presentes = []

for codigo, nome in [(0, "Normal"), (1, "Atenção"), (2, "Risco de Falha")]:
    qtd = contagem.get(codigo, 0)
    if qtd > 0:
        labels_presentes.append(f"{nome}\n({qtd} leituras)")
        valores_presentes.append(qtd)
        cores_presentes.append(cores_mapa[codigo])

plt.figure(figsize=(8, 6))
plt.pie(valores_presentes, labels=labels_presentes, colors=cores_presentes, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
plt.title("Distribuição das Previsões - Novos Dados (Cenário Real)\nPredictInd", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("grafico_resultado_uso.png", dpi=150)
plt.close()

# ----- GRÁFICO 2: Scatterplot Temperatura x Vibração -----
cores_pontos = [cores_mapa[int(c)] for c in predicoes_codigo]

plt.figure(figsize=(10, 6))
plt.scatter(df_resultado["temperatura"], df_resultado["vibracao"], c=cores_pontos, s=100, edgecolors="black", linewidths=0.5, alpha=0.85)

from matplotlib.patches import Patch
legenda = [Patch(facecolor="#2ecc71", edgecolor="black", label="Normal"), Patch(facecolor="#f39c12", edgecolor="black", label="Atenção"), Patch(facecolor="#e74c3c", edgecolor="black", label="Risco de Falha")]
plt.legend(handles=legenda, title="Status Previsto")

plt.title("Temperatura x Vibração — Classificação da IA (Com Overlap)\nPredictInd", fontsize=13, fontweight="bold")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Vibração (mm/s)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("grafico_temp_vibracao_classificado.png", dpi=150)
plt.close()

df_resultado.to_csv(ARQUIVO_SAIDA, index=False)
print("  ✔ Gráficos e CSV salvos com sucesso!")