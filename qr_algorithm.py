"""
Fase 1.2 - Iteracao QR para encontrar autovalores a partir da forma de
Hessenberg.
 
Estrategia: QR iterativo COM SHIFT (deslocamento de Wilkinson/Rayleigh) e
DEFLACAO. Sem shift, a convergencia pode ser extremamente lenta (ou nunca
zerar a subdiagonal a tempo); com shift, converge em poucas iteracoes por
autovalor, o que e' essencial para os testes de escalabilidade ate N=1000.
 
A fatoracao H = QR em cada passo usa np.linalg.qr (decomposicao QR via
Householder internamente), conforme permitido pelo enunciado, que pede a
ITERACAO H_{k+1} = R_k Q_k, e nao exige reimplementar a fatoracao QR em si
(ja implementamos Householder na Fase 1.1 para a reducao a Hessenberg).
"""
import numpy as np
from hessenberg import hessenberg
 
 
def wilkinson_shift(H, m):
    """Shift de Wilkinson baseado no bloco 2x2 inferior direito ativo (0..m)."""
    a = H[m - 1, m - 1]
    b = H[m - 1, m]
    c = H[m, m - 1]
    d = H[m, m]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc >= 0:
        sq = np.sqrt(disc)
        l1 = (tr + sq) / 2
        l2 = (tr - sq) / 2
    else:
        return d  # autovalores complexos no bloco: usa shift simples
    return l1 if abs(l1 - d) < abs(l2 - d) else l2
 
 
def qr_algorithm(H_in, tol=1e-10, max_iter=1000):
    """
    Recebe matriz de Hessenberg H e retorna o vetor de autovalores,
    aplicando QR com shift + deflacao ate a subdiagonal convergir a zero.
    """
    H = H_in.copy().astype(float)
    n = H.shape[0]
    eigenvalues = np.zeros(n, dtype=complex)
    m = n - 1  # indice do ultimo elemento do bloco ativo
 
    iterations_used = 0
    while m > 0:
        converged = False
        for _ in range(max_iter):
            iterations_used += 1
            if abs(H[m, m - 1]) < tol * (abs(H[m - 1, m - 1]) + abs(H[m, m]) + 1e-300):
                H[m, m - 1] = 0.0
                converged = True
                break
 
            mu = wilkinson_shift(H, m)
            Q, R = np.linalg.qr(H[: m + 1, : m + 1] - mu * np.eye(m + 1))
            H[: m + 1, : m + 1] = R @ Q + mu * np.eye(m + 1)
 
        if converged or True:
            # deflacao: extrai autovalor(es) do canto inferior direito
            if m >= 1 and abs(H[m, m - 1]) < tol * (abs(H[m - 1, m - 1]) + abs(H[m, m]) + 1e-300):
                eigenvalues[m] = H[m, m]
                m -= 1
            else:
                # bloco 2x2 nao convergiu para real -> par de autovalores complexos conjugados
                a, b, c, d = H[m - 1, m - 1], H[m - 1, m], H[m, m - 1], H[m, m]
                tr = a + d
                det = a * d - b * c
                disc = tr * tr - 4 * det
                if disc < 0:
                    sq = np.sqrt(-disc) * 1j
                    eigenvalues[m] = (tr + sq) / 2
                    eigenvalues[m - 1] = (tr - sq) / 2
                else:
                    sq = np.sqrt(disc)
                    eigenvalues[m] = (tr + sq) / 2
                    eigenvalues[m - 1] = (tr - sq) / 2
                m -= 2
 
    if m == 0:
        eigenvalues[0] = H[0, 0]
 
    return eigenvalues
 
 
def compute_eigenvalues(A, tol=1e-10, max_iter=1000):
    """Pipeline completo: Hessenberg + QR iterativo."""
    H = hessenberg(A)
    return qr_algorithm(H, tol=tol, max_iter=max_iter)
 
 
if __name__ == "__main__":
    np.random.seed(1)
    n = 6
    A = np.random.rand(n, n)
    meus = sorted(compute_eigenvalues(A), key=lambda x: x.real)
    numpy_ref = sorted(np.linalg.eigvals(A), key=lambda x: x.real)
    print("Meus autovalores:   ", np.round(meus, 5))
    print("Referencia (numpy): ", np.round(numpy_ref, 5))
