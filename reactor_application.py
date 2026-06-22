"""
Fase 4 - Dinamica de Redes de Reatores (CSTRs em serie).
 
Monta a matriz tridiagonal A (N=100) com:
- diagonal principal = -2.5
- sub/superdiagonal   = 1.0
 
Usa o algoritmo otimizado (Fase 3) para achar os autovalores, verifica
estabilidade (todas as partes reais negativas) e calcula a razao de
rigidez (Stiffness Ratio).
"""
import numpy as np
from qr_symmetric import symmetric_tridiagonal_eigenvalues
 
 
def montar_matriz_reatores(N, diag_principal=-2.5, fora_diagonal=1.0):
    d = np.full(N, diag_principal, dtype=float)
    e = np.full(N - 1, fora_diagonal, dtype=float)
    return d, e
 
 
def main():
    N = 100
    d, e = montar_matriz_reatores(N)
 
    autovalores = symmetric_tridiagonal_eigenvalues(d, e)
 
    print(f"--- Analise para N={N} reatores ---")
    print(f"Numero de autovalores encontrados: {len(autovalores)}")
    print(f"Maior autovalor: {autovalores.max():.6f}")
    print(f"Menor autovalor: {autovalores.min():.6f}")
 
    # 4.3 - Analise de estabilidade
    todas_negativas = np.all(autovalores < 0)
    print(f"\nTodas as partes reais sao negativas? {todas_negativas}")
    if todas_negativas:
        print(">> Sistema estavel: a concentracao tende a um estado estacionario.")
    else:
        print(">> Sistema instavel: ha risco de a concentracao divergir (explodir).")
 
    # 4.4 - razao de rigidez (Stiffness Ratio)
    parte_real_abs = np.abs(autovalores)
    S = parte_real_abs.max() / parte_real_abs.min()
    print(f"\nRazao de rigidez S = {S:.4f}")
    if S > 1e3:
        print(">> S > 10^3: sistema STIFF. Recomenda-se um solver implicito")
        print("   (ex: Backward Euler, BDF). Um solver explicito como RK4 exigiria")
        print("   passos de tempo extremamente pequenos para nao divergir,")
        print("   tornando a simulacao computacionalmente inviavel.")
    else:
        print(">> S <= 10^3: sistema nao rigido. Um solver explicito (ex: RK4)")
        print("   e' suficiente e mais barato computacionalmente.")
 
    np.savetxt("autovalores_reatores.csv", np.sort(autovalores), delimiter=",",
               header="autovalor", comments="")
 
 
if __name__ == "__main__":
    main()
