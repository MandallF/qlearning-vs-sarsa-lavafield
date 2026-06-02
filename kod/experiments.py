"""
Eğitim ve Deney Yardımcıları
============================

Bir ajanı ortamda eğitir, çok-tohumlu (multi-seed) deneyler çalıştırır ve
analiz için ödül/adım eğrilerini toplar. Ayrıca eğitilmiş bir ajanın açgözlü
(epsilon=0) politikasını değerlendiren `greedy_rollout` fonksiyonunu içerir.
"""

import numpy as np

from agents import TDAgent


def run_episode(env, agent):
    """Tek bir eğitim epizodu çalıştırır (epsilon-greedy davranış)."""
    s = env.reset()
    a = agent.select_action(s)
    total_reward = 0.0
    steps = 0
    while True:
        s_next, reward, terminated, truncated = env.step(a)
        # Sonraki eylem politikadan seçilir (SARSA bunu kullanır;
        # Q-Learning ve Expected SARSA için yalnızca bir sonraki adımdır)
        a_next = agent.select_action(s_next)
        agent.update(s, a, reward, s_next, a_next, terminated)
        s, a = s_next, a_next
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def train_agent(env, method, n_episodes, alpha, gamma, epsilon, seed):
    """Tek bir ajanı eğitir; ajanı ve epizod başına ödül/adım dizilerini döndürür."""
    rng = np.random.default_rng(seed)
    agent = TDAgent(env.n_states, env.n_actions, method=method,
                    alpha=alpha, gamma=gamma, epsilon=epsilon, rng=rng)
    rewards = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    for ep in range(n_episodes):
        rewards[ep], steps[ep] = run_episode(env, agent)
    return agent, rewards, steps


def run_experiment(env, methods, n_episodes=500, n_seeds=25,
                   alpha=0.5, gamma=0.95, epsilon=0.1):
    """
    Her yöntem için n_seeds bağımsız eğitim çalıştırır ve sonuçları ortalar.

    Döndürür: { method: { 'rewards_mean', 'rewards_std', 'steps_mean', 'agent' } }
    'agent', temsili (seed=0) eğitilmiş ajandır; politika görselleştirmesi için.
    """
    results = {}
    for method in methods:
        all_rewards = np.zeros((n_seeds, n_episodes))
        all_steps = np.zeros((n_seeds, n_episodes))
        representative = None
        for seed in range(n_seeds):
            agent, rewards, steps = train_agent(
                env, method, n_episodes, alpha, gamma, epsilon, seed)
            all_rewards[seed] = rewards
            all_steps[seed] = steps
            if seed == 0:
                representative = agent
        results[method] = {
            "rewards_mean": all_rewards.mean(axis=0),
            "rewards_std": all_rewards.std(axis=0),
            "steps_mean": all_steps.mean(axis=0),
            "agent": representative,
        }
    return results


def greedy_rollout(env, agent, max_steps=200):
    """
    Açgözlü (epsilon=0) politikayla tek bir epizod çalıştırır.
    Döndürür: (toplam_odul, adim_sayisi, yol)  -- yol: (satır,sütun) listesi.
    """
    s = env.reset()
    path = [env.state_to_pos(s)]
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        a = agent.best_action(s)
        s, reward, terminated, truncated = env.step(a)
        total += reward
        steps += 1
        path.append(env.state_to_pos(s))
        if terminated or truncated:
            break
    return total, steps, path
