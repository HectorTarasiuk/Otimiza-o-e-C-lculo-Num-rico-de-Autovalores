"""
Fase 1.1 - Reducao a Forma de Hessenberg via Transformacoes de Householder.
 
Ideia: para cada coluna k (de 0 a n-3), construimos um vetor de Householder
v que zera as entradas abaixo da subdiagonal naquela coluna. Aplicamos a
reflexao H_k = I - 2*v*v^T tanto pela esquerda quanto pela direita
(A = H_k A H_k), para manter os autovalores inalterados (transformacao de
similaridade ortogonal).
"""
import numpy as np
 
 
def householder_vector(x):
    """Constroi o vetor de Householder v tal que (I - 2vv^T)x = ||x|| e1."""
    v = x.copy().astype(float)
    alpha = -np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else -np.linalg.norm(x)
    v[0] = v[0] - alpha
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros_like(v)
    return v / norm_v
 
 
def hessenberg(A, symmetric=False):
    """
    Reduz a matriz densa A (n x n) a forma de Hessenberg superior H,
    usando reflexoes de Householder sucessivas.
 
    Se symmetric=True, a matriz resultante e automaticamente tridiagonal
    (pois A = A^T implica H = H^T, e Hessenberg simetrica = tridiagonal).
 
    Retorna H (matriz transformada).
    """
    n = A.shape[0]
    H = A.copy().astype(float)
 
    for k in range(n - 2):
        x = H[k + 1:, k]
        v = householder_vector(x)
        if not np.any(v):
            continue
 
        # reflexao pela esquerda: linhas k+1..n-1, colunas k..n-1
        H[k + 1:, k:] -= 2.0 * np.outer(v, v @ H[k + 1:, k:])
        # reflexao pela direita: colunas k+1..n-1, todas as linhas
        H[:, k + 1:] -= 2.0 * np.outer(H[:, k + 1:] @ v, v)
 
    if symmetric:
        # Limpa ruido numerico fora da banda tridiagonal (deveria ser ~0)
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    H[i, j] = 0.0
 
    return H
 
 
if __name__ == "__main__":
    # teste rapido, demonstracao do caso N=5
    np.random.seed(0)
    n = 5
    B = np.random.rand(n, n)
    A_sym = (B + B.T)  # matriz simetrica
 
    H_sym = hessenberg(A_sym, symmetric=True)
    print("Matriz simetrica original:\n", np.round(A_sym, 3))
    print("\nApos Householder (deve ser tridiagonal simetrica):\n", np.round(H_sym, 6))
 
    A_gen = np.random.rand(n, n)
    H_gen = hessenberg(A_gen)
    print("\nMatriz generica apos Householder (Hessenberg superior):\n", np.round(H_gen, 6))
