"""
Lav Tarlası Ortamı (LavaGridWorld)
==================================

Ajanın, bir lav şeridiyle bölünmüş ızgara biçimli bir tarlada başlangıç
hücresinden ödül bölgesine (hedef) ulaşmaya çalıştığı, tablolu pekiştirmeli
öğrenme için tasarlanmış ÖZEL bir GridWorld (ızgara dünyası) ortamıdır.

Ortam bir Markov Karar Süreci (MKS / MDP) olarak modellenmiştir:
    (S, A, P, R, gamma)
    S : durumlar kümesi (ızgaradaki konumlar)
    A : eylemler kümesi (Yukarı, Aşağı, Sol, Sağ)
    P : geçiş dinamiği (deterministik hareket; lava düşünce başlangıca dönüş)
    R : ödül fonksiyonu (adım, lav, tuzak, hedef)
    gamma : indirim faktörü (ajan tarafında kullanılır)

Tasarım amacı: Q-Learning'in "riskli ama kısa" (lav kenarı), SARSA'nın ise
"güvenli ama uzun" yolu öğrendiği klasik uçurum-yürüyüşü davranışını üreten,
ancak öğrenciye özgü bir ortam ortaya koymaktır.
"""

import numpy as np

# --- Hücre tipleri ---
EMPTY = 0   # boş hücre (normal adım)
START = 1   # başlangıç (B)
GOAL = 2    # hedef / ödül bölgesi (H)
LAVA = 3    # lav (L) -> büyük ceza + başlangıca dönüş
TRAP = 4    # tuzak (T) -> orta ceza, geçişe izin var

# --- Eylemler: 0=Yukarı, 1=Aşağı, 2=Sol, 3=Sağ ---
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Yukari", "Asagi", "Sol", "Sag"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]  # ↑ ↓ ← →

# --- Varsayılan ızgara yerleşimi (6 satır x 10 sütun) ---
#   .  : boş      S : başlangıç   G : hedef
#   L  : lav      T : tuzak
DEFAULT_LAYOUT = [
    "..........",
    "..T....T..",
    "..........",
    "....T.....",
    "..........",
    "SLLLLLLLLG",
]

CHAR_TO_TYPE = {".": EMPTY, "S": START, "G": GOAL, "L": LAVA, "T": TRAP}


class LavaGridWorld:
    """Lav Tarlası ızgara dünyası ortamı (MDP)."""

    def __init__(self, layout=None, step_reward=-1.0, lava_reward=-100.0,
                 trap_reward=-20.0, goal_reward=100.0, max_steps=500):
        layout = layout if layout is not None else DEFAULT_LAYOUT
        self.grid = np.array(
            [[CHAR_TO_TYPE[ch] for ch in row] for row in layout], dtype=int
        )
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTION_DELTAS)

        # Ödül parametreleri
        self.step_reward = step_reward
        self.lava_reward = lava_reward
        self.trap_reward = trap_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        # Özel konumlar
        self.start_pos = self._find(START)
        self.goal_pos = self._find(GOAL)

        self.agent_pos = self.start_pos
        self._steps = 0

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def _find(self, cell_type):
        rows, cols = np.where(self.grid == cell_type)
        return (int(rows[0]), int(cols[0]))

    def pos_to_state(self, pos):
        """(satır, sütun) -> tek sayı durum indeksi."""
        r, c = pos
        return r * self.n_cols + c

    def state_to_pos(self, state):
        """Durum indeksi -> (satır, sütun)."""
        return (state // self.n_cols, state % self.n_cols)

    def cell_type(self, pos):
        r, c = pos
        return int(self.grid[r, c])

    # ------------------------------------------------------------------
    # MDP arayüzü
    # ------------------------------------------------------------------
    def reset(self):
        """Ortamı başlangıç durumuna getirir, başlangıç durumunu döndürür."""
        self.agent_pos = self.start_pos
        self._steps = 0
        return self.pos_to_state(self.agent_pos)

    def step(self, action):
        """
        Bir eylem uygula.

        Döndürür: (next_state, reward, terminated, truncated)
          terminated : hedefe ulaşıldı mı (gerçek terminal durum)
          truncated  : adım sınırına ulaşıldı mı (yapay kesme)
        """
        self._steps += 1
        dr, dc = ACTION_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Izgara dışına çıkış -> duvar, yerinde kal
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            nr, nc = r, c

        cell = self.grid[nr, nc]
        terminated = False

        if cell == LAVA:
            # Lav: büyük ceza ve ajan tekrar başlangıca döner (epizod sürer)
            reward = self.lava_reward
            self.agent_pos = self.start_pos
        elif cell == GOAL:
            # Hedef/ödül bölgesi: pozitif ödül, epizod biter
            reward = self.goal_reward
            self.agent_pos = (nr, nc)
            terminated = True
        elif cell == TRAP:
            # Tuzak: orta ceza, ajan hücreye geçer
            reward = self.trap_reward
            self.agent_pos = (nr, nc)
        else:
            # Normal adım
            reward = self.step_reward
            self.agent_pos = (nr, nc)

        truncated = (self._steps >= self.max_steps) and not terminated
        return self.pos_to_state(self.agent_pos), reward, terminated, truncated

    # ------------------------------------------------------------------
    # Konsol gösterimi (hata ayıklama için)
    # ------------------------------------------------------------------
    def render(self):
        symbols = {EMPTY: ".", START: "B", GOAL: "H", LAVA: "L", TRAP: "T"}
        ar, ac = self.agent_pos
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == (ar, ac):
                    row.append("A")  # ajan
                else:
                    row.append(symbols[int(self.grid[r, c])])
            lines.append(" ".join(row))
        return "\n".join(lines)


if __name__ == "__main__":
    # Hızlı kendi kendine test
    env = LavaGridWorld()
    print("Ortam boyutu: %d x %d, durum=%d, eylem=%d"
          % (env.n_rows, env.n_cols, env.n_states, env.n_actions))
    print("Baslangic:", env.start_pos, "Hedef:", env.goal_pos)
    print(env.render())
