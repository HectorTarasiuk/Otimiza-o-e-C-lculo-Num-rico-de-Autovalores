(veja este arquivo em forma de código)
Otimização e Cálculo Numérico de Autovalores

Projeto da disciplina de Equações Diferenciais Ordinárias. Implementação do algoritmo padrão-ouro para autovalores
(Redução de Hessenberg + Iteração QR), estudo de escalabilidade e
otimização para matrizes simétricas, com aplicação a uma rede de reatores
químicos (CSTRs em série).

Estrutura


hessenberg.py — Fase 1.1: redução à forma de Hessenberg via
transformações de Householder (funciona para matriz genérica; quando
symmetric=True, a saída colapsa para tridiagonal).

qr_algorithm.py — Fase 1.2: iteração QR com shift de Wilkinson e
deflação, aplicada sobre a Hessenberg, para o caso geral.

qr_symmetric.py — Fase 3: versão otimizada para matrizes simétricas,
operando sobre a forma tridiagonal (menos memória, blocos ativos
encolhem a cada deflação).

scalability_study.py — Fases 2 e 3.3: gera matrizes aleatórias para
N ∈ {10, 50, 100, 250, 500, 1000}, mede tempo (via time.perf_counter)
e pico de memória (via tracemalloc), salva CSVs e gera os gráficos
grafico_tempo.png e grafico_memoria.png, além do ganho percentual
entre o algoritmo geral e o otimizado.

reactor_application.py — Fase 4: monta a matriz tridiagonal de
N=100 reatores, calcula autovalores com o algoritmo otimizado, verifica
estabilidade e calcula a razão de rigidez (Stiffness Ratio).

---Como rodar---
python -m venv venv
source venv/bin/activate        - Windows: venv\Scripts\activate
pip install -r requirements.txt
de maneira mais simplificada, criar pasta no visual studio, adicionar os segunte 5  arquivos que trminam com .py e tbm o requirements.txt, dps escrever no termnal pip install -r requirements.txt
e em seguida cada um dos 5 por vez

python hessenberg.py            - demonstração da Fase 1.1 (item 3.2, N=5)
python qr_algorithm.py          - valida o algoritmo geral contra numpy
python qr_symmetric.py          - valida o algoritmo otimizado contra numpy
python reactor_application.py   - Fase 4 completa

python scalability_study.py     - Fases 2 e 3.3 (demoraria muito tempo para N=1000, foi até 500 apenas)
                              
Observacoes de desempenho                        
O algoritmo geral (denso) reimplementado aqui é O(N³) por iteração da
fatoração QR, então cresce rápido para N grande 

Validacao
Cada módulo (hessenberg.py, qr_algorithm.py, qr_symmetric.py) tem um
bloco if __name__ == "__main__" que compara os autovalores calculados
com numpy.linalg.eigvals / eigvalsh como referência.
                                 
