"""
Ana Çalıştırma Dosyası
======================

Lav Tarlası ortamında Q-Learning ve SARSA yöntemlerini eğitir;
rastgele politika ve BFS (model-tabanlı en kısa yol) ile karşılaştırır; tüm
grafikleri üretir ve sayısal özet yazdırır/kaydeder.

Çalıştırmak için:  py main.py
"""

import os
import sys

import numpy as np

# Windows konsolunda Türkçe / ok karakterleri için UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from environment import LavaGridWorld
from experiments import run_experiment, greedy_rollout
from baselines import bfs_shortest_path, evaluate_random_policy
import visualize

# Çıktı klasörü: kod/ ile aynı seviyede figurler/
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figurler")

METHODS = ["q_learning", "sarsa"]
METHOD_LABELS = {
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
}

# Hiperparametreler (ders notlarındaki değerlere yakın)
N_EPISODES = 500
N_SEEDS = 25
ALPHA = 0.5
GAMMA = 0.95
EPSILON = 0.1


def main():
    env = LavaGridWorld()
    out = []

    def log(line=""):
        print(line)
        out.append(line)

    log("=" * 60)
    log("LAV TARLASI — Pekiştirmeli Öğrenme Deneyi")
    log("=" * 60)
    log("Ortam: %d x %d ızgara, %d durum, %d eylem"
        % (env.n_rows, env.n_cols, env.n_states, env.n_actions))
    log("Başlangıç: %s   Hedef: %s" % (env.start_pos, env.goal_pos))
    log("Parametreler: alpha=%.2f, gamma=%.2f, epsilon=%.2f, epizod=%d, seed=%d"
        % (ALPHA, GAMMA, EPSILON, N_EPISODES, N_SEEDS))
    log("")
    log("Ortam haritası:")
    log(env.render())
    log("")

    # --- Eğitim ---
    log("Eğitim çalışıyor (%d yöntem × %d seed × %d epizod)..."
        % (len(METHODS), N_SEEDS, N_EPISODES))
    results = run_experiment(env, METHODS, n_episodes=N_EPISODES,
                             n_seeds=N_SEEDS, alpha=ALPHA, gamma=GAMMA,
                             epsilon=EPSILON)

    # --- Temel çizgiler ---
    random_avg = evaluate_random_policy(env, n_episodes=300, seed=123)
    bfs_path = bfs_shortest_path(env)
    bfs_len = len(bfs_path) - 1 if bfs_path else -1

    # --- Grafikler ---
    os.makedirs(FIG_DIR, exist_ok=True)
    visualize.plot_environment(env, os.path.join(FIG_DIR, "01_ortam.png"))
    visualize.plot_learning_curves(results, METHOD_LABELS,
                                   os.path.join(FIG_DIR, "02_ogrenme_egrileri.png"))
    visualize.plot_steps(results, METHOD_LABELS,
                         os.path.join(FIG_DIR, "03_adim_sayisi.png"))
    visualize.plot_learned_paths(env, results, METHOD_LABELS,
                                 os.path.join(FIG_DIR, "04_ogrenilen_rotalar.png"))
    visualize.plot_comparison(env, results, METHOD_LABELS, random_avg,
                              os.path.join(FIG_DIR, "05_karsilastirma.png"))

    # --- Sayısal özet ---
    log("")
    log("-" * 60)
    log("SONUÇLAR")
    log("-" * 60)
    log("BFS (model-tabanlı) en kısa güvenli yol: %d adım" % bfs_len)
    log("Rastgele politika ortalama ödül: %.1f" % random_avg)
    log("")
    log("%-16s | %8s | %6s | %18s" %
        ("Yöntem", "Açgözlü", "Adım", "Son 50 ep. çevrimiçi"))
    log("%-16s-+-%8s-+-%6s-+-%18s" % ("-" * 16, "-" * 8, "-" * 6, "-" * 18))
    for m in METHODS:
        agent = results[m]["agent"]
        g_reward, g_steps, _ = greedy_rollout(env, agent)
        online_last = float(np.mean(results[m]["rewards_mean"][-50:]))
        log("%-16s | %8.0f | %6d | %18.1f"
            % (METHOD_LABELS[m], g_reward, g_steps, online_last))
    log("")
    log("Yorum: SARSA çevrimiçi (eğitim) ödülde genelde daha yüksektir (lav")
    log("kenarından kaçınıp daha az düşer); Q-Learning ise açgözlü politikada")
    log("en kısa (riskli) yolu bulur. Bu, on-policy/off-policy farkının somut")
    log("bir gösterimidir.")
    log("")
    log("Grafikler '%s' klasörüne kaydedildi." % os.path.normpath(FIG_DIR))

    # Özet metni kaydet (rapor için referans)
    with open(os.path.join(FIG_DIR, "..", "sonuclar.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
