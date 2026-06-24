# =============================================================
# PROJETO: PredictInd - Monitoramento Preditivo de Compressor
# ARQUIVO: gerar_dados.py
# DESCRIÇÃO: Gera os dados sintéticos de treinamento e uso da IA (Cenário Real)
# DISCIPLINA: Projeto Integrador Big Data na Indústria III
# =============================================================

import numpy as np
import pandas as pd
import os

# Define semente para reprodutibilidade dos dados gerados
np.random.seed(42)

# =============================================================
# FUNÇÃO: Gerar dados de treinamento (Cenário Desbalanceado + Overlap)
# =============================================================
def gerar_dados_treinamento(n_amostras=1000):
    dados = []
    
    # Probabilidades do mundo real: 80% Normal, 15% Atenção, 5% Risco
    classes = np.random.choice([0, 1, 2], size=n_amostras, p=[0.80, 0.15, 0.05])

    for classe in classes:
        if classe == 0: # NORMAL
            registro = {
                "temperatura"    : np.random.uniform(40, 75),
                "vibracao"       : np.random.uniform(0.5, 3.5),
                "corrente"       : np.random.uniform(10, 22),
                "tensao"         : np.random.uniform(210, 225),
                "rotacao"        : np.random.uniform(2750, 3000),
                "pressao"        : np.random.uniform(5.5, 8.0),
                "ruido"          : np.random.uniform(60, 78),
                "horas_operacao" : np.random.uniform(0, 1200),
                "carga_trabalho" : np.random.uniform(40, 75),
                "status_maquina" : 0
            }
        elif classe == 1: # ATENÇÃO
            registro = {
                "temperatura"    : np.random.uniform(70, 95),
                "vibracao"       : np.random.uniform(3.0, 5.5),
                "corrente"       : np.random.uniform(20, 27),
                "tensao"         : np.random.uniform(200, 215),
                "rotacao"        : np.random.uniform(2550, 2800),
                "pressao"        : np.random.uniform(4.5, 6.5),
                "ruido"          : np.random.uniform(74, 88),
                "horas_operacao" : np.random.uniform(1000, 2200),
                "carga_trabalho" : np.random.uniform(70, 88),
                "status_maquina" : 1
            }
        else: # RISCO DE FALHA
            registro = {
                "temperatura"    : np.random.uniform(90, 125),
                "vibracao"       : np.random.uniform(5.0, 11.0),
                "corrente"       : np.random.uniform(25, 38),
                "tensao"         : np.random.uniform(185, 205),
                "rotacao"        : np.random.uniform(2100, 2600),
                "pressao"        : np.random.uniform(1.5, 5.0),
                "ruido"          : np.random.uniform(85, 115),
                "horas_operacao" : np.random.uniform(2000, 4500),
                "carga_trabalho" : np.random.uniform(85, 100),
                "status_maquina" : 2
            }
        dados.append(registro)

    df = pd.DataFrame(dados)
    # Embaralha os dados
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.round(2)
    return df

# =============================================================
# FUNÇÃO: Gerar dados para uso (sem coluna de status)
# =============================================================
def gerar_dados_uso(n_amostras=100):
    dados = []
    classes = np.random.choice([0, 1, 2], size=n_amostras, p=[0.80, 0.15, 0.05])

    for classe in classes:
        if classe == 0:
            registro = {"temperatura": np.random.uniform(40, 75), "vibracao": np.random.uniform(0.5, 3.5), "corrente": np.random.uniform(10, 22), "tensao": np.random.uniform(210, 225), "rotacao": np.random.uniform(2750, 3000), "pressao": np.random.uniform(5.5, 8.0), "ruido": np.random.uniform(60, 78), "horas_operacao": np.random.uniform(0, 1200), "carga_trabalho": np.random.uniform(40, 75)}
        elif classe == 1:
            registro = {"temperatura": np.random.uniform(70, 95), "vibracao": np.random.uniform(3.0, 5.5), "corrente": np.random.uniform(20, 27), "tensao": np.random.uniform(200, 215), "rotacao": np.random.uniform(2550, 2800), "pressao": np.random.uniform(4.5, 6.5), "ruido": np.random.uniform(74, 88), "horas_operacao": np.random.uniform(1000, 2200), "carga_trabalho": np.random.uniform(70, 88)}
        else:
            registro = {"temperatura": np.random.uniform(90, 125), "vibracao": np.random.uniform(5.0, 11.0), "corrente": np.random.uniform(25, 38), "tensao": np.random.uniform(185, 205), "rotacao": np.random.uniform(2100, 2600), "pressao": np.random.uniform(1.5, 5.0), "ruido": np.random.uniform(85, 115), "horas_operacao": np.random.uniform(2000, 4500), "carga_trabalho": np.random.uniform(85, 100)}
        dados.append(registro)

    df = pd.DataFrame(dados)
    df = df.sample(frac=1, random_state=99).reset_index(drop=True).round(2)
    return df

# =============================================================
# EXECUÇÃO PRINCIPAL
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PredictInd - Gerador de Dados Sintéticos (Cenário Real)")
    print("=" * 60)

    print("\n[1/2] Gerando dados de TREINAMENTO...")
    df_treino = gerar_dados_treinamento(n_amostras=1000)
    df_treino.to_csv("dados_treinamento.csv", index=False)
    print(f"      ✔ Arquivo salvo: dados_treinamento.csv")
    print(f"      ✔ Total de registros: {len(df_treino)}")

    print("\n      Distribuição das classes:")
    contagem = df_treino["status_maquina"].value_counts().sort_index()
    for classe, qtd in contagem.items():
        nomes = {0: "Normal", 1: "Atenção", 2: "Risco de Falha"}
        print(f"        Classe {classe} ({nomes[classe]}): {qtd} registros")

    print("\n[2/2] Gerando dados de USO (novos dados dos sensores)...")
    df_uso = gerar_dados_uso(n_amostras=100)
    df_uso.to_csv("dados_uso.csv", index=False)
    print(f"      ✔ Arquivo salvo: dados_uso.csv")
    print(f"      ✔ Total de registros: {len(df_uso)}")

    print("\n" + "=" * 60)
    print("  Dados gerados com sucesso!")
    print("=" * 60)