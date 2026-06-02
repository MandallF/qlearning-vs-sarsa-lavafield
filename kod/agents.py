"""
Tablolu Zaman Farkı (TD) Ajanları
=================================

Tek bir `TDAgent` sınıfı üç farklı güncelleme kuralını destekler. Bunların
hepsi ders notlarında (5. hafta) verilen tablolu RL yöntemleridir:

  * Q-Learning (off-policy):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

  * SARSA (on-policy):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

  * Expected SARSA (on-policy beklenen değer):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * E_{a'~pi}[Q(s',a')] - Q(s,a) ]

Keşif-sömürü dengesi için epsilon-greedy politika kullanılır:
  olasılık (1 - epsilon) ile en iyi eylem, olasılık epsilon ile rastgele eylem.
"""

import numpy as np


class TDAgent:
    METHODS = ("q_learning", "sarsa", "expected_sarsa")

    def __init__(self, n_states, n_actions, method="q_learning",
                 alpha=0.5, gamma=0.95, epsilon=0.1, rng=None):
        assert method in self.METHODS, "Bilinmeyen yontem: %s" % method
        self.n_states = n_states
        self.n_actions = n_actions
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

        # Q tablosu: tüm değerler 0 ile başlatılır (ders notlarındaki gibi)
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    # ------------------------------------------------------------------
    # Eylem seçimi
    # ------------------------------------------------------------------
    def best_action(self, state):
        """Açgözlü eylem (deterministik; ilk en büyük). Görselleştirme/değerlendirme için."""
        return int(np.argmax(self.Q[state]))

    def _greedy_random_tie(self, state):
        """Açgözlü eylem; beraberlikte rastgele seç (eğitimde yanlılığı azaltır)."""
        q = self.Q[state]
        candidates = np.flatnonzero(q == q.max())
        return int(self.rng.choice(candidates))

    def select_action(self, state):
        """Epsilon-greedy eylem seçimi (davranış politikası)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_random_tie(state)

    def policy_probs(self, state):
        """Verilen durumda epsilon-greedy politikanın eylem olasılıkları."""
        q = self.Q[state]
        best_mask = (q == q.max())
        n_best = int(best_mask.sum())
        probs = np.full(self.n_actions, self.epsilon / self.n_actions)
        probs[best_mask] += (1.0 - self.epsilon) / n_best
        return probs

    # ------------------------------------------------------------------
    # Q güncellemesi
    # ------------------------------------------------------------------
    def update(self, s, a, r, s_next, a_next, terminated):
        if terminated:
            # Terminal durumda gelecekteki değer yoktur
            target = r
        elif self.method == "q_learning":
            target = r + self.gamma * self.Q[s_next].max()
        elif self.method == "sarsa":
            target = r + self.gamma * self.Q[s_next, a_next]
        else:  # expected_sarsa
            expected = float(np.dot(self.policy_probs(s_next), self.Q[s_next]))
            target = r + self.gamma * expected

        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
