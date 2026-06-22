"""
Fase 2 (geral) e Fase 3.3 (otimizado) - Estudo de escalabilidade
 
Mede tempo de execucao e pico de memoria RAM para encontrar todos os
autovalores de matrizes aleatorias de tamanhos crescentes
 
IMPORTANTE (leia antes de rodar):
- O algoritmo GERAL (denso) e' O(N^3) por iteracao do QR. Para N=500 e
  N=1000 ele pode demorar MUITO (minutos a dezenas de minutos), porque
  reimplementamos o algoritmo "na unha" em Python/numpy sem todas as
  otimizacoes de uma biblioteca como LAPACK.
- Se estiver sem tempo, rode primeiro com N_LIST_GERAL menor (ex: ate 250)
  e deixe N=500/1000 rodando em segundo plano, ou rode durante a noite.
- O algoritmo SIMETRICO/TRIDIAGONAL e' mais rapido (O(N^2)) e deve
  completar N=1000 em tempo bem mais razoavel.
 
Saida:
- scalability_general.csv
- scalability_symmetric.csv
- grafico_tempo.png
- grafico_memoria.png
"""
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from qr_algorithm import compute_eigenvalues
from qr_symmetric import compute_eigenvalues_symmetric
 
 
def medir(func, A):
    tracemalloc.start()
    t0 = time.perf_counter()
    func(A)
    t1 = time.perf_counter()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (t1 - t0), pico / (1024 ** 2)  
 
 
def estudo_geral(n_list, seed=42):
    rng = np.random.default_rng(seed)
    linhas = []
    for n in n_list:
        A = rng.random((n, n))
        print(f"[GERAL] N={n} rodando...")
        tempo, mem = medir(compute_eigenvalues, A)
        print(f"[GERAL] N={n}  tempo={tempo:.3f}s  memoria={mem:.2f}MB")
        linhas.append({"N": n, "tempo_s": tempo, "memoria_MB": mem, "metodo": "geral"})
    return pd.DataFrame(linhas)
 
 
def estudo_simetrico(n_list, seed=42):
    rng = np.random.default_rng(seed)
    linhas = []
    for n in n_list:
        B = rng.random((n, n))
        A = B + B.T
        print(f"[SIMETRICO] N={n} rodando...")
        tempo, mem = medir(compute_eigenvalues_symmetric, A)
        print(f"[SIMETRICO] N={n}  tempo={tempo:.3f}s  memoria={mem:.2f}MB")
        linhas.append({"N": n, "tempo_s": tempo, "memoria_MB": mem, "metodo": "simetrico"})
    return pd.DataFrame(linhas)
 
 
def plotar(df_geral, df_sim):
    fig, ax = plt.subplots()
    ax.plot(df_geral["N"], df_geral["tempo_s"], "o-", label="Geral (denso)")
    ax.plot(df_sim["N"], df_sim["tempo_s"], "o-", label="Otimizado (simetrico)")
    ax.set_xlabel("N (tamanho da matriz)")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Tempo de execucao vs N")
    ax.legend()
    fig.savefig("grafico_tempo.png", dpi=150, bbox_inches="tight")
 
    fig2, ax2 = plt.subplots()
    ax2.plot(df_geral["N"], df_geral["memoria_MB"], "o-", label="Geral (denso)")
    ax2.plot(df_sim["N"], df_sim["memoria_MB"], "o-", label="Otimizado (simetrico)")
    ax2.set_xlabel("N (tamanho da matriz)")
    ax2.set_ylabel("Memoria de pico (MB)")
    ax2.set_title("Memoria vs N")
    ax2.legend()
    fig2.savefig("grafico_memoria.png", dpi=150, bbox_inches="tight")
 
 
if __name__ == "__main__":
    # a lista geral teve que ser limitada a N=500, porque N=1000 demoraria muito tempo para rodar
    N_LIST_GERAL = [10, 50, 100, 250, 500]
    N_LIST_SIM = [10, 50, 100, 250, 500, 1000]
 
    df_geral = estudo_geral(N_LIST_GERAL)
    df_geral.to_csv("scalability_general.csv", index=False)
 
    df_sim = estudo_simetrico(N_LIST_SIM)
    df_sim.to_csv("scalability_symmetric.csv", index=False)
 
    plotar(df_geral, df_sim)
 
    # Ganho percentual de desempenho
    comp = df_geral.merge(df_sim, on="N", suffixes=("_geral", "_sim"))
    comp["ganho_tempo_%"] = 100 * (comp["tempo_s_geral"] - comp["tempo_s_sim"]) / comp["tempo_s_geral"]
    comp["ganho_memoria_%"] = 100 * (comp["memoria_MB_geral"] - comp["memoria_MB_sim"]) / comp["memoria_MB_geral"]
    comp.to_csv("comparativo_ganho.csv", index=False)
    print("\nResumo do ganho percentual:")
    print(comp[["N", "ganho_tempo_%", "ganho_memoria_%"]])
