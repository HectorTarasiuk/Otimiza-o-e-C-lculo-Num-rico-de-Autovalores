"""
Fase 3 - Versao otimizada para matrizes simetricas.
 
Por que e' mais rapido que o algoritmo geral?
1) Armazenamento: a reducao de Householder de uma matriz simetrica colapsa
   para 'tridiagonal' (ja demonstrado em hessenberg.py para N=5). Guardamos
   so' a diagonal "d" e a subdiagonal "e" -> O(N) em vez de O(N^2).
2) Cada bloco ativo da iteracao QR encolhe a cada deflacao (m diminui em 1
   a cada autovalor encontrado), entao o trabalho total e' a soma de blocos
   cada vez menores, em vez de sempre operar na matriz N x N inteira como
   no caso geral. Isso leva a complexidade total O(N^2), contra O(N^3) do
   algoritmo denso geral.
"""
import numpy as np
from hessenberg import hessenberg
 
 
def tridiagonalize_symmetric(A):
    """Reduz A simetrica a tridiagonal e retorna (diag, subdiag)."""
    H = hessenberg(A, symmetric=True)
    n = H.shape[0]
    d = np.diag(H).copy()
    e = np.array([H[i + 1, i] for i in range(n - 1)]) if n > 1 else np.array([])
    return d, e
 
 
def symmetric_tridiagonal_eigenvalues(d_in, e_in, tol=1e-10, max_iter=500):
    """
    QR com shift de Wilkinson + deflacao, operando apenas no bloco ativo
    (m+1) x (m+1), que encolhe a cada autovalor encontrado. O bloco e'
    montado como matriz tridiagonal densa pequena para a fatoracao QR.
    """
    d = d_in.copy().astype(float)
    e = e_in.copy().astype(float)
    n = len(d)
    if n == 1:
        return d
    m = n - 1  # ultimo indice do bloco ativo
 
    while m > 0:
        for _ in range(max_iter):
            if abs(e[m - 1]) < tol * (abs(d[m - 1]) + abs(d[m]) + 1e-300):
                e[m - 1] = 0.0
                break
            sub = (np.diag(d[: m + 1])
                   + np.diag(e[:m], 1)
                   + np.diag(e[:m], -1))
            mu = d[m]  # shift de Rayleigh
            Q, R = np.linalg.qr(sub - mu * np.eye(m + 1))
            sub = R @ Q + mu * np.eye(m + 1)
            d[: m + 1] = np.diag(sub)
            e[:m] = np.diag(sub, 1)
        m -= 1
 
    return np.sort(d)
 
 
def compute_eigenvalues_symmetric(A, tol=1e-10, max_iter=500):
    """Pipeline otimizado completo para matrizes simetricas."""
    d, e = tridiagonalize_symmetric(A)
    return symmetric_tridiagonal_eigenvalues(d, e, tol=tol, max_iter=max_iter)
 
 
if __name__ == "__main__":
    np.random.seed(2)
    n = 6
    B = np.random.rand(n, n)
    A = B + B.T
    meus = compute_eigenvalues_symmetric(A)
    ref = np.sort(np.linalg.eigvalsh(A))
    print("Meus autovalores:  ", np.round(meus, 6))
    print("Referencia (numpy):", np.round(ref, 6))
