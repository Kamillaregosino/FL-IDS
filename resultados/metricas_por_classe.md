# Métricas por classe: centralizado × federado

Conjunto de teste: 30.295 conexões (25%), avaliado no servidor. Dados normalizados, semente 42.

- **Centralizado:** 100 épocas, B = 16
- **Federado:** Nt = 20, N = 5, B = 16, 100 rodadas, FedAvg; E = 15 (melhor configuração do artigo) e E = 20 (nossa melhor)

Arquivos de origem: `resultados/relatorio_*.txt`, `resultados/matriz_*.csv` e `resultados/matriz_*.png`.

## F1-score por classe

| Classe (exemplos no teste) | Centralizado | Federado E=15 | Federado E=20 |
|---|---|---|---|
| normal (10.000) | 0,9919 | 0,9918 | 0,9905 |
| dos (10.000) | 0,9922 | 0,9945 | 0,9963 |
| probe (10.000) | 0,9912 | 0,9933 | 0,9932 |
| r2l (282) | 0,8231 | 0,8213 | 0,8284 |
| u2r (13) | 0,0000 | 0,0000 | 0,0000 |
| **Acurácia** | 98,99% | 99,13% | 99,15% |
| **F1 macro** | 0,7597 | 0,7602 | 0,7617 |

## Precisão e recall das classes raras

| Classe | Métrica | Centralizado | Federado E=15 | Federado E=20 |
|---|---|---|---|---|
| r2l | precisão | 0,7709 | 0,7967 | 0,8102 |
| r2l | recall | 0,8830 | 0,8475 | 0,8475 |
| u2r | precisão | 0,0000 | 0,0000 | 0,0000 |
| u2r | recall | 0,0000 | 0,0000 | 0,0000 |

## Matrizes de confusão

Linhas = classe real, colunas = classe prevista. Ordem: normal, dos, probe, r2l, u2r.

**Centralizado**
```
[[9912    7    9   72    0]
 [   7 9931   62    0    0]
 [  21   81 9896    2    0]
 [  33    0    0  249    0]
 [  13    0    0    0    0]]
```

**Federado E=15**
```
[[9925    8    8   59    0]
 [   7 9968   25    0    0]
 [  27   71 9900    2    0]
 [  42    0    1  239    0]
 [  13    0    0    0    0]]
```

**Federado E=20**
```
[[9936    3    7   54    0]
 [   8 9969   23    0    0]
 [  63   41 9894    2    0]
 [  43    0    0  239    0]
 [  13    0    0    0    0]]
```

## O que os resultados mostram

1. **O federado não perde para o centralizado, nem nas classes raras.** O F1 por classe é praticamente igual nas três execuções; em dos e probe, o federado fica até um pouco acima. Isso confirma o resultado do artigo, agora olhando classe por classe.
2. **Nenhum modelo detecta u2r.** Os 13 exemplos de teste foram classificados como normal nos três casos. É uma limitação dos dados (só 39 exemplos de u2r no treino, espalhados entre 20 clientes, cerca de 2 por cliente) e do modelo pequeno (16 neurônios), não do aprendizado federado em si.
3. **r2l é detectado em cerca de 85% dos casos.** Os erros vão quase todos para normal, e entre 54 e 72 conexões normais são marcadas como r2l (falso alarme).
4. **A acurácia esconde o problema.** Ela fica perto de 99% porque normal, dos e probe somam 99% do teste. O F1 macro, que dá o mesmo peso a cada classe, fica em torno de 0,76. O artigo original reporta só acurácia.
5. **Reprodutibilidade.** As acurácias desta execução diferem no máximo 0,02 ponto das da Tabela 2 (98,99 × 98,97; 99,13 × 99,11; 99,15 × 99,17), com o mesmo código e a mesma semente. A diferença provavelmente vem da ordem de execução paralela dos clientes no Ray e da máquina.

## Possíveis trabalhos futuros

- Pesos por classe na função de perda ou reamostragem de r2l/u2r para tentar detectar u2r.
- Particionamento não IID entre clientes, para ver como as classes raras se comportam quando ficam concentradas em poucos clientes.
