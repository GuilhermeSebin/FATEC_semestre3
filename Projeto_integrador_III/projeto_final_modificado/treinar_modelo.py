# =============================================================
# PROJETO: PredictInd - Monitoramento Preditivo de Compressor
# ARQUIVO: treinar_modelo.py
# DESCRIÇÃO: Treina a Rede Neural e salva o modelo treinado
# DISCIPLINA: Projeto Integrador Big Data na Indústria III
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ARQUIVO_DADOS = "dados_treinamento.csv"
ARQUIVO_MODELO   = "modelo_manutencao.pkl"
ARQUIVO_SCALER   = "scaler_manutencao.pkl"
ARQUIVO_RESULTADO = "resultado_treinamento.txt"

CLASSES = {0: "Normal", 1: "Atenção", 2: "Risco de Falha"}

print("=" * 60)
print("  PredictInd - Treinamento do Modelo de IA")
print("=" * 60)

print("\n[PASSO 1] Carregando dados de treinamento...")
if not os.path.exists(ARQUIVO_DADOS):
    print(f"\n  ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado!")
    exit(1)

df = pd.read_csv(ARQUIVO_DADOS)
print(f"  ✔ Dados carregados com sucesso! Total: {len(df)}")

print("\n[PASSO 2] Separando variáveis de entrada e variável alvo...")
X = df.drop("status_maquina", axis=1)
y = df["status_maquina"]

print("\n[PASSO 3] Dividindo dados em Treino (80%) e Teste (20%)...")
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[PASSO 4] Normalizando os dados com StandardScaler...")
scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm  = scaler.transform(X_teste)
joblib.dump(scaler, ARQUIVO_SCALER)

print("\n[PASSO 5] Treinando a Rede Neural (MLPClassifier)...")
modelo = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=800, # AUMENTADO PARA LIDAR COM DADOS COMPLEXOS/OVERLAP
    random_state=42,
    verbose=False
)
modelo.fit(X_treino_norm, y_treino)
joblib.dump(modelo, ARQUIVO_MODELO)
print(f"  ✔ Modelo treinado! Épocas: {modelo.n_iter_} | Loss: {modelo.loss_:.6f}")

print("\n[PASSO 6] Avaliando o modelo...")
y_previsto = modelo.predict(X_teste_norm)
acuracia = accuracy_score(y_teste, y_previsto)
relatorio = classification_report(y_teste, y_previsto, target_names=["Normal", "Atenção", "Risco de Falha"])
matriz_conf = confusion_matrix(y_teste, y_previsto)

print(f"\n  ACURÁCIA DO MODELO: {acuracia*100:.2f}%")
print(matriz_conf)

print("\n[PASSO 7] Gerando gráficos...")
# ----- GRÁFICO 1: Matriz de Confusão -----
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Atenção", "Risco de Falha"], yticklabels=["Normal", "Atenção", "Risco de Falha"])
plt.title("Matriz de Confusão - Treinamento do Modelo\nPredictInd - Compressor Industrial", fontsize=13, fontweight="bold")
plt.ylabel("Valor Real", fontsize=11)
plt.xlabel("Valor Previsto pela IA", fontsize=11)
plt.tight_layout()
plt.savefig("grafico_matriz_confusao_treino.png", dpi=150)
plt.close()

# ----- GRÁFICO 2: Curva de Loss -----
plt.figure(figsize=(9, 5))
plt.plot(modelo.loss_curve_, color="steelblue", linewidth=2)
plt.title("Curva de Loss durante o Treinamento\nPredictInd - Rede Neural MLPClassifier", fontsize=13, fontweight="bold")
plt.xlabel("Épocas de Treinamento", fontsize=11)
plt.ylabel("Loss (Erro)", fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("grafico_curva_loss.png", dpi=150)
plt.close()

# ----- GRÁFICO 3: Distribuição das classes -----
contagem_classes = pd.Series(y_treino).value_counts().sort_index()
cores = ["#2ecc71", "#f39c12", "#e74c3c"]
nomes_classes = ["Normal", "Atenção", "Risco de Falha"]

plt.figure(figsize=(8, 5))
barras = plt.bar(nomes_classes, contagem_classes.values, color=cores, edgecolor="black")
plt.title("Distribuição Desbalanceada - Conjunto de Treino\nPredictInd", fontsize=13, fontweight="bold")
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width() / 2., altura + 1, f'{int(altura)}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("grafico_distribuicao_classes.png", dpi=150)
plt.close()

print("\n[PASSO 8] Salvando resultados em arquivo de texto...")
with open(ARQUIVO_RESULTADO, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n  PredictInd - Resultado do Treinamento\n" + "=" * 60 + "\n\n")
    f.write(f"  Acurácia: {acuracia*100:.2f}%\n\n")
    f.write(relatorio + "\n\n" + str(matriz_conf))

print("\n  TREINAMENTO CONCLUÍDO COM SUCESSO!")