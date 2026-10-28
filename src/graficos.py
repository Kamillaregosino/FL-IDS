"""
Uso:  python src/graficos.py [--normalizar]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from comum import RESULTADOS 

p = argparse.ArgumentParser()
p.add_argument("--normalizar", action="store_true")
a = p.parse_args()
pre = "norm" if a.normalizar else "bruto"
RES = RESULTADOS

# Valores publicados no artigo 
ARTIGO_CENTRAL = 99.16
ARTIGO = {
    (5, 5, 16): 98.35, (5, 10, 16): 99.33, (5, 15, 16): 99.40, (5, 20, 16): 99.17,
    (5, 1, 16): 97.70, (10, 1, 16): 97.37, (15, 1, 16): 96.74, (20, 1, 16): 96.62,
    (5, 15, 32): 98.83, (5, 15, 64): 99.02, (5, 15, 128): 98.15, (5, 15, 256): 98.02,
}
GRUPOS = [  # (figura, título, parâmetro variado, configurações)
    ("figura5", "Variando N (E=1, B=16)", "N", [(n, 1, 16) for n in (5, 10, 15, 20)]),
    ("figura6", "Variando E (N=5, B=16)", "E", [(5, e, 16) for e in (5, 10, 15, 20)]),
    ("figura7", "Variando B (N=5, E=15)", "B", [(5, 15, b) for b in (32, 64, 128, 256)]),
]


def ler(nome):
    caminho = os.path.join(RES, nome)
    if not os.path.exists(caminho):
        return None
    with open(caminho) as f:
        return [(int(r[0]), float(r[1])) for r in list(csv.reader(f))[1:]]


central = ler(f"centralizado_{pre}_B16.csv")
acc_central = central[-1][1] * 100 if central else None

if central:
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*central), color="#3A5BA0")
    plt.axhline(ARTIGO_CENTRAL / 100, ls="--", color="gray", label=f"artigo: {ARTIGO_CENTRAL}%")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (teste)")
    plt.title(f"Figura 4 — Centralizado (final: {acc_central:.2f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, f"figura4_{pre}.png"), dpi=130)
    plt.close()

linhas_tabela = []
for fig, titulo, var, configs in GRUPOS:
    plt.figure(figsize=(7, 4))
    for cfg in configs:
        curva = ler(f"fl_{pre}_N{cfg[0]}_E{cfg[1]}_B{cfg[2]}.csv")
        if not curva or curva[-1][0] < 100:
            continue
        valor = dict(zip("NEB", cfg))[var]
        plt.plot(*zip(*curva), label=f"{var}={valor}")
        acc = curva[-1][1] * 100
        linhas_tabela.append((cfg, acc))
    if acc_central:
        plt.axhline(acc_central / 100, ls="--", color="gray", label="centralizado")
    plt.ylim(0.8, 1.0)
    plt.xlabel("Rodada de comunicação")
    plt.ylabel("Acurácia do modelo global")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, f"{fig}_{pre}.png"), dpi=130)
    plt.close()

with open(os.path.join(RES, f"tabela2_{pre}.md"), "w") as f:
    f.write("| Nt | N | E | B | Artigo (%) | DIF artigo | Replicação (%) | DIF replicação |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    if acc_central:
        f.write(f"| — | — | 100 épocas | 16 | {ARTIGO_CENTRAL:.2f} | — | {acc_central:.2f} | — |\n")
    for (n, e, b), acc in linhas_tabela:
        art = ARTIGO[(n, e, b)]
        dif = f"{acc - acc_central:+.2f}" if acc_central else "—"
        f.write(f"| 20 | {n} | {e} | {b} | {art:.2f} | {art - ARTIGO_CENTRAL:+.2f} | {acc:.2f} | {dif} |\n")
print(open(os.path.join(RES, f"tabela2_{pre}.md")).read())
