"""
Karşılaştırma Temel Çizgileri (Baselines)
=========================================

RL yöntemlerini "başka bir yöntemle" karşılaştırmak için iki referans:

  1) Rastgele politika : her adımda eşit olasılıkla rastgele eylem. Öğrenmenin
     kattığı değeri göstermek için alt sınır (alt referans) görevi görür.

  2) BFS (Genişlik Öncelikli Arama) : ortamın modeli BİLİNİYORSA, lavdan
     kaçınan en kısa yolu klasik (model-tabanlı) planlama ile bulur. RL'in
     model bilmeden öğrendiği çözümle karşılaştırmak için üst referans.
"""

from collections import deque
import numpy as np

from environment import LAVA, ACTION_DELTAS


def bfs_shortest_path(env):
    """
    Başlangıçtan hedefe, lav hücrelerinden kaçınan en kısa yolu döndürür.
    Yol, (satır, sütun) konumlarının listesidir (başlangıç ve hedef dahil).
    Bu, model BİLİNİYORSA elde edilebilecek optimal (en kısa) güvenli yoldur.
    """
    start, goal = env.start_pos, env.goal_pos
    frontier = deque([start])
    came_from = {start: None}

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        r, c = current
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
                continue
            nxt = (nr, nc)
            if nxt in came_from:
                continue
            if env.grid[nr, nc] == LAVA:
                continue  # lav engel kabul edilir
            came_from[nxt] = current
            frontier.append(nxt)

    if goal not in came_from:
        return []

    # Yolu geriye doğru izle
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def evaluate_random_policy(env, n_episodes=300, seed=0):
    """Rastgele politikanın epizod başına ortalama toplam ödülünü döndürür."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_episodes):
        env.reset()
        total = 0.0
        while True:
            action = int(rng.integers(env.n_actions))
            _, reward, terminated, truncated = env.step(action)
            total += reward
            if terminated or truncated:
                break
        totals.append(total)
    return float(np.mean(totals))
