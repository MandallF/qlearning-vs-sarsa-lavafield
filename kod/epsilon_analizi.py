"""
Epsilon (Keşif Oranı) Duyarlılık Analizi
========================================

ε (keşif oranı) değerini değiştirmenin etkisini inceler. Farklı ε değerleri için
Q-Learning ve SARSA'nın:
  * çevrimiçi (eğitim) ödülünü ve
  * epizod başına adım sayısını
ölçer (her biri çok-tohumlu ortalama).

Beklenen sonuç: ε büyüdükçe ajan daha sık rastgele hareket eder, lavaya daha çok
düşer; bu yüzden çevrimiçi ödül DÜŞER ve adım sayısı ARTAR. Buna karşılık öğrenilen
açgözlü (ε=0) politika büyük ölçüde optimal kalır.

Çalıştırmak için:  py epsilon_analizi.py
"""

import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import LavaGridWorld
from experiments import train_agent, greedy_rollout
from visualize import METHOD_COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figurler")

EPS_LIST = [0.01, 0.05, 0.1, 0.2, 0.3]
METHODS = ["q_learning", "sarsa"]
LABELS = {"q_learning": "Q-Learning", "sarsa": "SARSA"}
N_EPISODES = 300
N_SEEDS = 12
ALPHA, GAMMA = 0.5, 0.95


def main():
    env = LavaGridWorld()
    res = {m: {"online_r": [], "online_s": [], "greedy_r": []} for m in METHODS}

    print("Epsilon duyarlılık analizi (%d yöntem × %d ε × %d tohum × %d epizod)..."
          % (len(METHODS), len(EPS_LIST), N_SEEDS, N_EPISODES))
    for m in METHODS:
        for eps in EPS_LIST:
            o_r, o_s, g_r = [], [], []
            for seed in range(N_SEEDS):
                agent, rewards, steps = train_agent(
                    env, m, N_EPISODES, ALPHA, GAMMA, eps, seed)
                o_r.append(float(rewards[-50:].mean()))
                o_s.append(float(steps[-50:].mean()))
                g_r.append(greedy_rollout(env, agent)[0])
            res[m]["online_r"].append(np.mean(o_r))
            res[m]["online_s"].append(np.mean(o_s))
            res[m]["greedy_r"].append(np.median(g_r))

    # --- Figür 08: iki panel ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in METHODS:
        ax1.plot(EPS_LIST, res[m]["online_r"], "o-", color=METHOD_COLORS[m],
                 linewidth=2, label=LABELS[m])
    ax1.set_xlabel("Keşif oranı ε")
    ax1.set_ylabel("Çevrimiçi ödül (son 50 epizod ort.)")
    ax1.set_title("ε Arttıkça Çevrimiçi (Eğitim) Ödülü Düşer")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for m in METHODS:
        ax2.plot(EPS_LIST, res[m]["online_s"], "s-", color=METHOD_COLORS[m],
                 linewidth=2, label=LABELS[m])
    ax2.set_xlabel("Keşif oranı ε")
    ax2.set_ylabel("Epizod başına adım (son 50 epizod ort.)")
    ax2.set_title("ε Arttıkça Adım Sayısı Artar")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("Epsilon (Keşif Oranı) Duyarlılık Analizi", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "08_epsilon.png"), dpi=150)
    plt.close(fig)

    # --- Özet tablo ---
    print("\n" + "=" * 64)
    print("EPSILON DUYARLILIK ÖZETİ (%d tohum ortalaması)" % N_SEEDS)
    print("=" * 64)
    for m in METHODS:
        print("\n%s:" % LABELS[m])
        print("   %6s | %14s | %12s | %12s" %
              ("ε", "çevrimiçi ödül", "çevr. adım", "açgözlü ödül"))
        for i, eps in enumerate(EPS_LIST):
            print("   %6.2f | %14.1f | %12.1f | %12.0f" %
                  (eps, res[m]["online_r"][i], res[m]["online_s"][i],
                   res[m]["greedy_r"][i]))
    print("\nFigür: 08_epsilon.png")


if __name__ == "__main__":
    main()
