# =============================================================
# PROJETO: PredictInd - Monitoramento Preditivo de Compressor
# ARQUIVO: testar_modelo.py
# DESCRIÇÃO: Carrega o modelo treinado e avalia seu desempenho (Gráficos)
# DISCIPLINA: Projeto Integrador Big Data na Indústria III
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ARQUIVO_DADOS    = "dados_treinamento.csv"
ARQUIVO_MODELO   = "modelo_manutencao.pkl"
ARQUIVO_SCALER   = "scaler_manutencao.pkl"
ARQUIVO_RESULTADO = "resultado_teste.txt"

print("=" * 60)
print("  PredictInd - Teste do Modelo de IA")
print("=" * 60)

print("\n[PASSO 1-3] Carregando e Preparando dados...")
modelo = joblib.load(ARQUIVO_MODELO)
scaler = joblib.load(ARQUIVO_SCALER)
df = pd.read_csv(ARQUIVO_DADOS)

X = df.drop("status_maquina", axis=1)
y = df["status_maquina"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_teste_norm = scaler.transform(X_teste)

print("\n[PASSO 4] Executando previsões...")
y_previsto = modelo.predict(X_teste_norm)

print("\n[PASSO 6] Calculando métricas de avaliação...")
acuracia  = accuracy_score(y_teste, y_previsto)
relatorio = classification_report(y_teste, y_previsto, target_names=["Normal", "Atenção", "Risco de Falha"])
matriz    = confusion_matrix(y_teste, y_previsto)

print(f"\n  Acurácia  : {acuracia*100:>6.2f}%")
print(relatorio)

print("\n[PASSO 7] Gerando gráfico da matriz de confusão...")
plt.figure(figsize=(9, 7))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Oranges", xticklabels=["Normal", "Atenção", "Risco de Falha"], yticklabels=["Normal", "Atenção", "Risco de Falha"])
plt.title(f"Matriz de Confusão - Conjunto de Teste\nPredictInd | Acurácia: {acuracia*100:.2f}%", fontsize=13, fontweight="bold")
plt.ylabel("Classe Real", fontsize=11)
plt.xlabel("Classe Prevista pela IA", fontsize=11)
plt.tight_layout()
plt.savefig("grafico_matriz_confusao_teste.png", dpi=150)
plt.close()

# Gráfico de Barras de Métricas
relatorio_dict = classification_report(y_teste, y_previsto, target_names=["Normal", "Atenção", "Risco de Falha"], output_dict=True)
classes_nomes = ["Normal", "Atenção", "Risco de Falha"]
precisao_por_classe = [relatorio_dict[c]["precision"] for c in classes_nomes]
recall_por_classe   = [relatorio_dict[c]["recall"]    for c in classes_nomes]
f1_por_classe       = [relatorio_dict[c]["f1-score"]  for c in classes_nomes]

x = np.arange(len(classes_nomes))
largura = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - largura, precisao_por_classe, largura, label="Precisão", color="#3498db", edgecolor="black")
b2 = ax.bar(x,            recall_por_classe,   largura, label="Recall", color="#2ecc71", edgecolor="black")
b3 = ax.bar(x + largura, f1_por_classe,        largura, label="F1-Score", color="#e67e22", edgecolor="black")

ax.set_title("Métricas de Desempenho por Classe\nPredictInd", fontsize=13, fontweight="bold")
ax.set_ylabel("Pontuação (0 a 1)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(classes_nomes)
ax.legend()
plt.tight_layout()
plt.savefig("grafico_metricas_por_classe.png", dpi=150)
plt.close()

print("\n[PASSO 8] Salvando resultados em arquivo...")
with open(ARQUIVO_RESULTADO, "w", encoding="utf-8") as f:
    f.write(f"Acurácia: {acuracia*100:.2f}%\n\n{relatorio}\n\n{matriz}")

print("\n  TESTE CONCLUÍDO COM SUCESSO!")