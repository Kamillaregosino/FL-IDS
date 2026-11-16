# FL-IDS — aprendizado federado para detecção de intrusão

Replicação do artigo **“Abordagem baseada em Aprendizado Federado para a Detecção de Intrusão em
Redes de Computadores”** (Damaceno, Ribeiro, Oliveira-Jr e Oliveira — UFG, 2023).

Uma rede neural classifica conexões do KDD Cup 99 em `normal` ou num de 4 tipos de ataque
(`dos`, `probe`, `r2l`, `u2r`). O treino é federado: 20 clientes, cada um com a sua parte dos
dados, e um servidor que só recebe os pesos e os combina com FedAvg (Flower). Comparamos com um
modelo centralizado, como no artigo, e repetimos as 12 configurações da Tabela 2.

## Estrutura

```
FL-IDS/
├── dataset/              KDD Cup 99: bruto, metadados, preproc.py e o CSV processado
├── src/
│   ├── comum.py          dados (27 atributos, 75/25, 20 partições iguais) e o modelo
│   ├── centralizado.py   referência: 100 épocas
│   ├── federado.py       servidor: Flower + FedAvg, Nt=20, N sorteados por rodada
│   ├── cliente.py        cliente: recebe o modelo global, treina E épocas, devolve os pesos
│   └── graficos.py       Figuras 4–7 e Tabela 2 comparadas com o artigo
├── scripts/
│   └── rodar_tabela2.sh  as 12 configurações em sequência (pula as já concluídas)
├── docker/               o mesmo experimento com 1 contêiner servidor + 20 clientes
├── resultados/           curvas (.csv), figuras e tabela
└── logs/                 saída de cada execução e tabela2_progresso.log
```

O modelo é uma MLP `27 → 16 (ReLU) → 5 (softmax)`, com 533 pesos e Adam.
O treino (75%) é embaralhado e dividido em 20 partes de 4.544 ou 4.545 conexões, sem
sobreposição. O teste (25%) fica só no servidor.

## Como rodar

Sempre a partir da raiz do repositório. Testado em Ubuntu 20.04 com Python 3.11.9 e sem GPU.

```bash
# 1. Ambiente (~2,3 GB em disco)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Referência centralizada (~3 h cada em CPU de 4 threads)
.venv/bin/python src/centralizado.py --normalizar
.venv/bin/python src/centralizado.py

# 3. Tabela 2: 12 configurações (de 6 a 45 min cada)
scripts/rodar_tabela2.sh --normalizar

# 4. Figuras e tabela comparadas com o artigo
.venv/bin/python src/graficos.py --normalizar
```

Uma configuração isolada: `.venv/bin/python src/federado.py --N 5 --E 1 --B 16 --rodadas 100 --normalizar`.

Execuções longas:

- `PARALELOS` controla quantos clientes treinam ao mesmo tempo. O padrão do script é 3;
  com 5, 11 GiB de RAM não bastam. Cada cliente usa cerca de 770 MB.
- Ao final de cada configuração, o script informa quantas rodadas perderam clientes. Qualquer
  valor diferente de 0 invalida o resultado.
- Se o computador suspender, o Ray perde os clientes e a execução cai. Para bloquear a suspensão
  automática: `systemd-inhibit --what=idle:sleep scripts/rodar_tabela2.sh --normalizar`.

Para regenerar o CSV processado a partir do KDD bruto (usa 2–3 GB de RAM):
`cd dataset && ../.venv/bin/python preproc.py`.

### Docker: 1 servidor + 20 clientes em contêineres

Cada cliente roda no seu próprio contêiner, só com a sua partição, e
conversa com o servidor pela rede (gRPC, porta 8080). O modelo, as partições e a semente são os
mesmos da simulação. 

**Pré-requisitos**

- Docker com Compose v2: `docker --version` e `docker compose version` devem responder.
  Se aparecer `permission denied` no `docker.sock`, use `sudo` ou entre no grupo `docker`.
- O ambiente `.venv` da raiz, só para gerar os dados (passo 1).
- Cerca de 2 GB livres em disco para a imagem (`python:3.11-slim` + TensorFlow + Flower).

**1. Gerar os dados de cada contêiner e o `docker-compose.yml`**

```bash
cd docker
../.venv/bin/python gerar.py --N 5 --E 1 --B 16 --rodadas 2 --normalizar
```

Cria `dados/teste.npz` (só o servidor recebe), `dados/cliente_0.npz` … `cliente_19.npz` (um por
contêiner) e o `docker-compose.yml` com N, E, B e o número de rodadas. Comece com `--rodadas 2`
para um teste rápido. Opções: `--memoria` (padrão `400m`) e `--cpus` (padrão `0.5`) por cliente.

**2. Construir a imagem e subir os 21 contêineres**

```bash
docker compose up --build -d
docker compose ps
```

O `--build` refaz a imagem com o código atual. É obrigatório na primeira vez e depois de qualquer
mudança em `servidor.py`, `cliente.py` ou `modelo.py`. O `docker compose ps` deve listar `servidor`
e `cliente0` … `cliente19` em execução. Nos primeiros segundos um cliente pode cair por ainda
não achar o servidor; o `restart: on-failure` o reinicia sozinho.

**3. Acompanhar as rodadas**

```bash
docker compose logs -f servidor
```

O servidor imprime `rodada   0  acurácia global …` (modelo inicial) e depois uma linha por rodada.
`Ctrl+C` só sai do log, não para os contêineres. Quando a última rodada aparece, o servidor encerra.

**4. Guardar o resultado e desligar tudo**

```bash
docker compose logs --no-color servidor > ../logs/docker_N5_E1_B16_R2.log
docker compose down
```

O servidor do Docker não grava CSV: a curva de acurácia fica só nesse log.

**Como saber se deu certo**

- `docker compose ps -a`: o `servidor` termina como `Exited (0)`.
- O log do servidor chega a `rodada   2` (ou à última rodada pedida) e cada rodada mostra
  `received 5 results and 0 failures`. Qualquer falha invalida o resultado, como na simulação.
- Nenhum cliente morto por falta de memória:
  `docker inspect --format '{{.Name}} {{.State.OOMKilled}}' $(docker compose ps -aq)` deve
  mostrar `false` para todos. Se algum der `true`, gere de novo com `--memoria 600m`.

**5. Experimento completo**

Repita os passos 1 a 4 com `--rodadas 100` e a configuração desejada (por exemplo
`--N 10 --E 1 --B 16`). 

## Escolhas onde o artigo não falou

| Lacuna | Escolha | Por quê |
|---|---|---|
| Taxa de aprendizado | 0,001 (padrão do Adam) | o artigo não muda o padrão |
| Normalização | `sklearn.preprocessing.normalize` | sem ela o centralizado oscila (até 74% numa época); com ela fica a 0,19 ponto do artigo |
| Batch do centralizado | 16 | o mesmo da configuração base do federado |
| E nos testes de B | 15 | valor da Tabela 2 (o texto do artigo diz 20) |
| Divisão dos dados | 75/25; treino embaralhado e dividido em 20 partes iguais | o texto é ambíguo; assim federado e centralizado usam o mesmo teste |
| Semente | 42, fixa | reprodutibilidade |

## Versões

Python 3.11.9 · tensorflow-cpu 2.21.0 · keras 3.15.1 · flwr 1.40.0 · ray 2.55.1 ·
scikit-learn 1.9.1 · pandas 3.0.6 · numpy 2.4.6 · matplotlib 3.11.2.
Lista completa em `requirements.lock.txt`.


O `federado.py` usa `flwr.simulation.start_simulation`, marcada como obsoleta no Flower 1.40.0: funciona, mas gera avisos.
