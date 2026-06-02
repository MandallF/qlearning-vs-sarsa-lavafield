"""
Görselleştirme
==============

Tüm grafikleri üretip PNG olarak kaydeder. Ekran gerektirmeyen 'Agg' arka
ucu kullanılır; bu yüzden sunucu/komut satırı ortamında da çalışır.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from environment import EMPTY, START, GOAL, LAVA, TRAP, ACTION_DELTAS
from experiments import greedy_rollout

# Hücre tipi -> renk  (EMPTY, START, GOAL, LAVA, TRAP)
CELL_COLORS = ["#eceff1", "#1e88e5", "#43a047", "#e53935", "#8e24aa"]
CELL_CMAP = ListedColormap(CELL_COLORS)
CELL_LABELS = {START: "B", GOAL: "H", TRAP: "T"}

# Yöntem renkleri (tüm grafiklerde tutarlı olması için)
METHOD_COLORS = {
    "q_learning": "#1f77b4",      # mavi
    "sarsa": "#ff7f0e",           # turuncu
    "expected_sarsa": "#2ca02c",  # yeşil
}


def _smooth(x, window=20):
    """Basit hareketli ortalama ile yumuşatma."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _draw_grid(ax, env):
    """Izgarayı renkli hücreler ve etiketlerle çizer."""
    ax.imshow(env.grid, cmap=CELL_CMAP, vmin=0, vmax=4)
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            t = int(env.grid[r, c])
            if t in CELL_LABELS:
                ax.text(c, r, CELL_LABELS[t], ha="center", va="center",
                        color="white", fontweight="bold", fontsize=11)


def plot_environment(env, path):
    """Ortam şemasını (B / H / lav / tuzak) açıklamalı çizer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_grid(ax, env)
    ax.set_title("Lav Tarlası Ortamı (6 × 10 ızgara)", fontsize=13)
    legend_items = [
        Patch(facecolor=CELL_COLORS[START], label="Başlangıç (B)"),
        Patch(facecolor=CELL_COLORS[GOAL], label="Hedef / Ödül (H)"),
        Patch(facecolor=CELL_COLORS[LAVA], label="Lav (−100, başa dön)"),
        Patch(facecolor=CELL_COLORS[TRAP], label="Tuzak (−20)"),
        Patch(facecolor=CELL_COLORS[EMPTY], label="Boş (−1)"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(results, labels, path):
    """Epizod başına toplam ödülün öğrenme eğrileri (yumuşatılmış, seed-ortalama)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, res in results.items():
        y = _smooth(res["rewards_mean"])
        ax.plot(np.arange(len(y)), y, label=labels[method], linewidth=2,
                color=METHOD_COLORS.get(method))
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Epizod başına toplam ödül (yumuşatılmış)")
    ax.set_title("Öğrenme Eğrileri — Çevrimiçi (eğitim) Performansı")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_steps(results, labels, path):
    """Epizod başına adım sayısı (yakınsama ve yol uzunluğu göstergesi)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, res in results.items():
        y = _smooth(res["steps_mean"])
        ax.plot(np.arange(len(y)), y, label=labels[method], linewidth=2,
                color=METHOD_COLORS.get(method))
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Epizod başına adım sayısı (yumuşatılmış)")
    ax.set_title("Epizod Başına Adım Sayısı")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learned_paths(env, results, labels, path,
                       methods_to_show=("q_learning", "sarsa")):
    """
    Seçili yöntemlerin açgözlü politikasını (oklar) ve izlenen rotayı
    (vurgulu çizgi) yan yana ızgaralarda gösterir. Projenin "vurucu" görseli:
    Q-Learning riskli (lav kenarı) yolu, SARSA güvenli yolu öğrenir.
    """
    n = len(methods_to_show)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, method in zip(axes, methods_to_show):
        _draw_grid(ax, env)
        agent = results[method]["agent"]
        # Açgözlü politikayı oklarla göster (lav ve hedef hariç)
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                t = int(env.grid[r, c])
                if t in (LAVA, GOAL):
                    continue
                a = agent.best_action(env.pos_to_state((r, c)))
                dr, dc = ACTION_DELTAS[a]
                ax.arrow(c, r, dc * 0.28, dr * 0.28, head_width=0.16,
                         head_length=0.16, fc="black", ec="black",
                         length_includes_head=True)
        # İzlenen açgözlü rotayı vurgula
        total, steps, rollout = greedy_rollout(env, agent)
        ys = [p[0] for p in rollout]
        xs = [p[1] for p in rollout]
        ax.plot(xs, ys, color="#00e5ff", linewidth=3.5, alpha=0.85,
                solid_capstyle="round")
        ax.set_title("%s\n(açgözlü ödül = %.0f, adım = %d)"
                     % (labels[method], total, steps), fontsize=12)
    fig.suptitle("Öğrenilen Açgözlü Politikalar ve İzlenen Rotalar",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(env, results, labels, random_avg, path):
    """
    Yöntemlerin açgözlü epizod ödüllerini + rastgele temel çizgiyi karşılaştıran
    çubuk grafik. Geniş değer aralığı için symlog ölçeği kullanılır.
    """
    names, values, colors = [], [], []
    for method in results.keys():
        total, _, _ = greedy_rollout(env, results[method]["agent"])
        names.append(labels[method])
        values.append(total)
        colors.append(METHOD_COLORS.get(method, "#1976d2"))
    names.append("Rastgele")
    values.append(random_avg)
    colors.append("#757575")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_yscale("symlog")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Açgözlü epizod ödülü (symlog ölçeği)")
    ax.set_title("Yöntem Karşılaştırması — Açgözlü Performans")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v,
                "%.0f" % v, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
