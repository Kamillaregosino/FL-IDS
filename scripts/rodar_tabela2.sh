#!/usr/bin/env bash
# Roda as 12 configurações da Tabela 2 do artigo (Nt = 20 clientes, 100 rodadas cada).
# Uso:  scripts/rodar_tabela2.sh [--normalizar]
# Variáveis: PY = Python do ambiente (padrão .venv/bin/python);
#            PARALELOS = clientes treinando ao mesmo tempo (padrão 3; com 5, 11 GiB de RAM não bastam).
# Logs em logs/, curvas em resultados/. Configurações já concluídas são puladas.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
PARALELOS="${PARALELOS:-3}"
EXTRA="${1:-}"
mkdir -p logs

rodar() {  # N E B
  local log="logs/fl${EXTRA:+_norm}_N$1_E$2_B$3.log"
  if grep -q "rodada 100" "$log" 2>/dev/null; then
    echo "já concluída: $log"; return
  fi
  echo "$(date +%H:%M) iniciando N=$1 E=$2 B=$3 (paralelos=$PARALELOS)"
  $PY src/federado.py --N "$1" --E "$2" --B "$3" --rodadas 100 --paralelos "$PARALELOS" $EXTRA > "$log" 2>&1
  # rodada em que o Ray perdeu clientes (ex.: falta de memória) = resultado inválido
  local falhas
  falhas=$(grep -a "aggregate_fit: received" "$log" | grep -v -c " 0 failures")
  echo "$(date +%H:%M) fim N=$1 E=$2 B=$3: $(grep 'rodada 100' "$log" || echo ERRO) · rodadas com clientes perdidos: $falhas"
}

# Figura 5: varia N (E=1, B=16)
for N in 5 10 15 20; do rodar "$N" 1 16; done
# Figura 7: varia B (N=5, E=15 — valor da Tabela 2; o texto diz E=20)
for B in 32 64 128 256; do rodar 5 15 "$B"; done
# Figura 6: varia E (N=5, B=16)
for E in 5 10 15 20; do rodar 5 "$E" 16; done
