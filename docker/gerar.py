"""Separa os dados por contêiner e gera o docker-compose.yml (1 servidor + 20 clientes).

Uso:  python gerar.py --N 5 --E 1 --B 16 --rodadas 100 [--normalizar]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from comum import NT, carregar  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=5)
p.add_argument("--E", type=int, default=1)
p.add_argument("--B", type=int, default=16)
p.add_argument("--rodadas", type=int, default=100)
p.add_argument("--normalizar", action="store_true")
p.add_argument("--memoria", default="400m", help="limite de memória de cada cliente")
p.add_argument("--cpus", default="0.5", help="limite de CPU de cada cliente")
a = p.parse_args()

aqui = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(aqui, "dados"), exist_ok=True)
d = carregar(a.normalizar)
np.savez_compressed(os.path.join(aqui, "dados", "teste.npz"), X=d["X_test"], y=d["y_test"])
for i in range(NT):
    idx = d[f"idx_{i}"]
    np.savez_compressed(os.path.join(aqui, "dados", f"cliente_{i}.npz"),
                        X=d["X_train"][idx], y=d["y_train"][idx])

linhas = [
    "# Gerado por gerar.py — não edite à mão",
    "x-base: &base",
    "  build: .",
    "  image: fl-ids-artigo",
    "",
    "services:",
    "  servidor:",
    "    <<: *base",
    "    command: python servidor.py",
    f"    environment: {{N: {a.N}, NT: {NT}, RODADAS: {a.rodadas}}}",
    "    volumes: [./dados/teste.npz:/dados/teste.npz:ro]",
]
for i in range(NT):
    linhas += [
        f"  cliente{i}:",
        "    <<: *base",
        "    command: python cliente.py",
        f"    environment: {{CID: {i}, E: {a.E}, B: {a.B}}}",
        f"    volumes: [./dados/cliente_{i}.npz:/dados/particao.npz:ro]",
        f"    mem_limit: {a.memoria}",
        f"    cpus: {a.cpus}",
        "    depends_on: [servidor]",
        "    restart: on-failure",
    ]
with open(os.path.join(aqui, "docker-compose.yml"), "w") as f:
    f.write("\n".join(linhas) + "\n")
print(f"docker-compose.yml gerado: N={a.N} E={a.E} B={a.B} rodadas={a.rodadas}")
