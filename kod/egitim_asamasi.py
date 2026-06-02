"""
Eğitim Aşaması Analizi
======================

Q-Learning ve SARSA'nın eğitim boyunca AÇGÖZLÜ (epsilon=0) politikasının nasıl
geliştiğini gösterir:

  * Belirli epizod kontrol noktalarında öğrenilen rota anlık görüntüleri,
  * Açgözlü performansın epizoda göre yakınsama eğrisi ve politikanın kaçıncı
    epizodda kararlı hâle geldiği.

Mevcut modülleri (environment, agents, experiments, visualize) yeniden kullanır.
Çalıştırmak için:  py egitim_asamasi.py
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
from agents import TDAgent
from experiments import run_episode, greedy_rollout
from visualize import _draw_grid, METHOD_COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figurler")

ALPHA, GAMMA, EPSILON = 0.5, 0.95, 0.1
N_EPISODES = 500
N_SEEDS = 20
CHECKPOINTS = [0, 5, 15, 40, 120, 500]
LABELS = {"q_learning": "Q-Learning", "sarsa": "SARSA"}


def train_track(env, method, n_episodes, seed, checkpoints):
    """Bir ajanı eğitirken her epizod sonunda açgözlü politikayı değerlendirir."""
    rng = np.random.default_rng(seed)
    agent = TDAgent(env.n_states, env.n_actions, method=method,
                    alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, rng=rng)
    g_reward = np.zeros(n_episodes + 1)
    g_steps = np.zeros(n_episodes + 1)
    snaps = {}
    r, s, path = greedy_rollout(env, agent)        # epizod 0: eğitimsiz
    g_reward[0], g_steps[0] = r, s
    if 0 in checkpoints:
        snaps[0] = (r, s, path)
    for ep in range(1, n_episodes + 1):
        run_episode(env, agent)
        r, s, path = greedy_rollout(env, agent)
        g_reward[ep], g_steps[ep] = r, s
        if ep in checkpoints:
            snaps[ep] = (r, s, path)
    return agent, g_reward, g_steps, snaps


def convergence_ep(mean_reward, tol):
    """Ortalama açgözlü ödülün, son değerinin ±tol bandında kalıp bir daha
    çıkmadığı ilk epizodu döndürür."""
    final = mean_reward[-1]
    for ep in range(len(mean_reward)):
        if np.all(np.abs(mean_reward[ep:] - final) <= tol):
            return ep, final
    return len(mean_reward) - 1, final


def draw_route(ax, env, path, title):
    _draw_grid(ax, env)
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, color="#00e5ff", linewidth=3, alpha=0.85, solid_capstyle="round")
    ax.set_title(title, fontsize=11)


def main():
    env = LavaGridWorld()
    os.makedirs(FIG_DIR, exist_ok=True)
    methods = ["q_learning", "sarsa"]

    mean_curves = {}
    snaps_by_method = {}
    print("Eğitim aşaması analizi çalışıyor (%d yöntem × %d tohum × %d epizod)..."
          % (len(methods), N_SEEDS, N_EPISODES))
    for method in methods:
        all_r = np.zeros((N_SEEDS, N_EPISODES + 1))
        snaps0 = None
        for seed in range(N_SEEDS):
            _, gr, _, snaps = train_track(env, method, N_EPISODES, seed, CHECKPOINTS)
            all_r[seed] = gr
            if seed == 0:
                snaps0 = snaps
        mean_curves[method] = np.median(all_r, axis=0)
        snaps_by_method[method] = snaps0

    # --- Figür 06: Q-Learning eğitim aşaması anlık görüntüleri ---
    snaps = snaps_by_method["q_learning"]
    cps = sorted(snaps.keys())
    ncol, nrow = 3, int(np.ceil(len(cps) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, cp in enumerate(cps):
        r, s, path = snaps[cp]
        et = "Eğitimsiz (0. epizod)" if cp == 0 else "%d. epizod" % cp
        draw_route(axes[i], env, path,
                   "%s\naçgözlü ödül=%.0f, adım=%d" % (et, r, s))
    for j in range(len(cps), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Q-Learning: Eğitim İlerledikçe Öğrenilen Açgözlü Rota",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5)
    fig.savefig(os.path.join(FIG_DIR, "06_egitim_asamalari.png"), dpi=150)
    plt.close(fig)

    # --- Figür 07: yakınsama eğrisi ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    conv_info = {}
    txt = []
    for method in methods:
        mc = mean_curves[method]
        ax.plot(np.arange(len(mc)), mc, label=LABELS[method],
                color=METHOD_COLORS[method], linewidth=2)
        ce, final = convergence_ep(mc, tol=5.0)
        conv_info[method] = (ce, final)
        ax.axvline(ce, color=METHOD_COLORS[method], linestyle="--", alpha=0.55)
        txt.append("%s: ~%d. epizod" % (LABELS[method], ce))
    ax.text(0.97, 0.42,
            "Kararlı hâle gelme:\n" + "\n".join(txt),
            transform=ax.transAxes, ha="right", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Açgözlü politika ödülü (%d tohum ortancası)" % N_SEEDS)
    ax.set_title("Eğitim Yakınsaması: Açgözlü Performansın Epizoda Göre Değişimi")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_yakinsama.png"), dpi=150)
    plt.close(fig)

    # --- Metinsel özet ---
    print("\n" + "=" * 60)
    print("EĞİTİM AŞAMASI ÖZETİ (%d tohum ortancası)" % N_SEEDS)
    print("=" * 60)
    for method in methods:
        mc = mean_curves[method]
        ce, final = conv_info[method]
        print("\n%s — açgözlü ödülün epizoda göre değişimi:" % LABELS[method])
        for ep in [0, 5, 15, 40, 120, 500]:
            print("   %4d. epizod: %6.1f" % (ep, mc[ep]))
        print("   >>> ~%d. epizodda KARARLI hâle geldi (son değer %.0f, ±5 bandı)"
              % (ce, final))
    print("\nFigürler: 06_egitim_asamalari.png, 07_yakinsama.png")


if __name__ == "__main__":
    main()
