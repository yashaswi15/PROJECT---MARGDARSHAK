import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import torch
torch._dynamo.config.suppress_errors = True

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(r"D:\projects__\resonate_26\work")
from config import (
    RL_TIMESTEPS, RL_LEARNING_RATE,
    RL_N_STEPS, RL_BATCH_SIZE, MODEL_DIR
)


# ─────────────────────────────────────────
#  ACTION SPACE
# ─────────────────────────────────────────
ACTION_TYPES = [
    "do_nothing",
    "open_drainage",
    "reroute_edge",
    "reduce_signal",
    "green_corridor",
    "emergency_drain",
]
N_ACTIONS     = len(ACTION_TYPES)
N_TARGETS     = 5
TOTAL_ACTIONS = N_ACTIONS * N_TARGETS   # 30

# 49-dim observation — compact, MLP-friendly
OBS_DIM = N_TARGETS * 6 + N_TARGETS * 3 + 4


def _obs(arr) -> np.ndarray:
    """Contiguous float64 array — what SB3 needs."""
    return np.ascontiguousarray(arr, dtype=np.float64)


# ─────────────────────────────────────────
#  CITY ENVIRONMENT
# ─────────────────────────────────────────
class CityEnv(gym.Env):
    """
    Gymnasium environment wrapping TwinEngine.
    Agent sees compact 49-dim risk summary.
    Reward penalises original problem + any
    cascade the agent creates solving it.
    """
    metadata = {"render_modes": []}

    def __init__(self, twin, max_steps: int = 200):
        super().__init__()
        self.twin      = twin
        self.max_steps = max_steps
        self._step     = 0

        self._top_risk_nodes: List[int]   = []
        self._top_cong_edges: List[Tuple] = []

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float64,
        )
        self.action_space    = spaces.Discrete(TOTAL_ACTIONS)
        self._prev_summary   = {}
        self.action_log: List[Dict] = []

    # ── GYMNASIUM API ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.twin.reset()
        self._step      = 0
        self.action_log = []

        self.twin.set_rainfall(float(np.random.uniform(0, 70)))
        self.twin.set_traffic_surge(float(np.random.uniform(1.0, 3.5)))

        for _ in range(5):
            self.twin.step()

        self._update_targets()
        self._prev_summary = self.twin._compute_summary()
        return _obs(self._build_obs()), {}

    def step(self, action: int):
        self._step += 1

        action_type_idx = int(action) % N_ACTIONS
        target_idx      = int(action) // N_ACTIONS
        action_name     = ACTION_TYPES[action_type_idx]
        cost            = self._apply_action(action_name, target_idx)

        self.twin.step()
        self._update_targets()

        new_summary = self.twin._compute_summary()
        reward      = self._compute_reward(self._prev_summary, new_summary, cost)
        self._prev_summary = new_summary

        terminated = self._step >= self.max_steps

        self.action_log.append({
            "step":     self._step,
            "action":   action_name,
            "reward":   round(float(reward), 4),
            "avg_risk": round(float(new_summary.get("avg_risk_score", 0)), 4),
        })

        return _obs(self._build_obs()), float(reward), terminated, False, {}

    # ── OBSERVATION ──────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        node_feats = np.zeros((N_TARGETS, 6), dtype=np.float64)
        for i, nid in enumerate(self._top_risk_nodes[:N_TARGETS]):
            if nid in self.twin.nodes:
                s = self.twin.nodes[nid]
                node_feats[i] = [
                    float(s.water_level),
                    float(s.traffic_density),
                    float(min(s.aqi / 300.0, 1.0)),
                    float(s.flood_risk),
                    float(s.congestion_risk),
                    float(s.risk_score),
                ]

        edge_feats = np.zeros((N_TARGETS, 3), dtype=np.float64)
        for i, ek in enumerate(self._top_cong_edges[:N_TARGETS]):
            if ek in self.twin.edges:
                e = self.twin.edges[ek]
                edge_feats[i] = [
                    float(e.flow_ratio()),
                    float(min(e.pollution_corridor / 300.0, 1.0)),
                    0.0 if e.flood_passable else 1.0,
                ]

        summary = self.twin._compute_summary()
        n       = max(summary.get("total_zones", 1), 1)
        global_feats = np.array([
            float(summary.get("avg_risk_score", 0)),
            float(min(summary.get("avg_aqi", 0) / 300.0, 1.0)),
            float(summary.get("flooded_zones", 0)) / n,
            float(summary.get("critical_zones", 0)) / n,
        ], dtype=np.float64)

        return np.concatenate([
            node_feats.flatten(),
            edge_feats.flatten(),
            global_feats,
        ])

    def _get_obs(self) -> np.ndarray:
        return _obs(self._build_obs())

    # ── ACTION ───────────────────────────────────────────────────

    def _apply_action(self, action_type: str, target_idx: int) -> float:
        if action_type == "do_nothing":
            return 0.0
        elif action_type == "open_drainage":
            if target_idx < len(self._top_risk_nodes):
                self.twin.apply_action("open_drainage",
                    self._top_risk_nodes[target_idx], value=0.5)
                return 0.1
        elif action_type == "reroute_edge":
            if target_idx < len(self._top_cong_edges):
                self.twin.apply_action("reroute_edge",
                    self._top_cong_edges[target_idx], value=0.6)
                return 0.15
        elif action_type == "reduce_signal":
            if target_idx < len(self._top_risk_nodes):
                self.twin.apply_action("reduce_signal",
                    self._top_risk_nodes[target_idx], value=1.0)
                return 0.1
        elif action_type == "green_corridor":
            if target_idx < len(self._top_cong_edges):
                self.twin.apply_action("green_corridor",
                    [self._top_cong_edges[target_idx]], value=0.7)
                return 0.05
        elif action_type == "emergency_drain":
            if target_idx < len(self._top_risk_nodes):
                self.twin.apply_action("open_drainage",
                    self._top_risk_nodes[target_idx], value=1.0)
                return 0.3
        return 0.0

    # ── REWARD ───────────────────────────────────────────────────

    def _compute_reward(self, prev: Dict, curr: Dict, cost: float) -> float:
        n = max(curr.get("total_zones", 1), 1)

        flood_imp    = (prev.get("flooded_zones", 0)  - curr.get("flooded_zones", 0))  / n
        critical_imp = (prev.get("critical_zones", 0) - curr.get("critical_zones", 0)) / n
        risk_imp     =  prev.get("avg_risk_score", 0) - curr.get("avg_risk_score", 0)
        aqi_imp      = (prev.get("avg_aqi", 0)        - curr.get("avg_aqi", 0))        / 300.0

        flood_worse = max(curr.get("flooded_zones", 0)  - prev.get("flooded_zones", 0),  0) / n
        crit_worse  = max(curr.get("critical_zones", 0) - prev.get("critical_zones", 0), 0) / n
        cascade     = (flood_worse + crit_worse) * 2.0

        reward = (
            flood_imp    * 3.0 +
            critical_imp * 2.0 +
            risk_imp     * 1.5 +
            aqi_imp      * 1.0 -
            cost         * 0.5 -
            cascade            -
            0.01
        )
        return float(np.clip(reward, -5.0, 5.0))

    # ── HELPERS ──────────────────────────────────────────────────

    def _update_targets(self):
        self._top_risk_nodes = sorted(
            self.twin.nodes.keys(),
            key=lambda nid: self.twin.nodes[nid].risk_score,
            reverse=True,
        )[:N_TARGETS * 4]

        self._top_cong_edges = sorted(
            self.twin.edges.keys(),
            key=lambda ek: self.twin.edges[ek].flow_ratio(),
            reverse=True,
        )[:N_TARGETS * 4]

    def get_action_summary(self) -> Dict:
        if not self.action_log:
            return {}
        counts = {}
        for e in self.action_log:
            counts[e["action"]] = counts.get(e["action"], 0) + 1
        return {
            "total_steps":   self._step,
            "action_counts": counts,
            "last_reward":   self.action_log[-1]["reward"],
            "last_risk":     self.action_log[-1]["avg_risk"],
        }


# ─────────────────────────────────────────
#  TRAINING CALLBACK
# ─────────────────────────────────────────
class TrainingCallback(BaseCallback):
    def __init__(self, log_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_every        = log_every
        self.episode_rewards: List[float] = []
        self._ep_reward       = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals.get("rewards", [0])[0])
        if self.num_timesteps % self.log_every == 0:
            avg = float(np.mean(self.episode_rewards[-20:])) \
                  if self.episode_rewards else 0.0
            print(f"  Step {self.num_timesteps:6d} | avg reward: {avg:.4f}",
                  end="\r")
        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True

    def _on_training_end(self):
        print()


# ─────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────
def train_agent(twin, timesteps: int = RL_TIMESTEPS) -> PPO:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ RLAgent ] Training PPO on {device}")
    print(f"  Obs dim    : {OBS_DIM}")
    print(f"  Action dim : {TOTAL_ACTIONS}")

    env      = CityEnv(twin)
    callback = TrainingCallback(log_every=500)

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = RL_LEARNING_RATE,
        n_steps       = RL_N_STEPS,
        batch_size    = RL_BATCH_SIZE,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        verbose       = 0,
        device        = device,
    )

    model.learn(
        total_timesteps = timesteps,
        callback        = callback,
        progress_bar    = False,
    )
    print(f"[ RLAgent ] Training complete.")
    return model


def save_agent(model: PPO, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[ RLAgent ] Saved → {path}")


def load_agent(path: str, twin) -> PPO:
    model = PPO.load(path, env=CityEnv(twin))
    print(f"[ RLAgent ] Loaded ← {path}")
    return model


# ─────────────────────────────────────────
#  INFERENCE WRAPPER
# ─────────────────────────────────────────
class SelfHealingAgent:
    """
    Live inference — called by FastAPI every twin.step()
    when AI is enabled. Returns action dict for
    operator console display.
    """

    def __init__(self, model: PPO, env: CityEnv):
        self.model  = model
        self.env    = env
        self._steps = 0

    def act(self) -> Dict:
        obs       = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        action    = int(action)

        action_type_idx = action % N_ACTIONS
        target_idx      = action // N_ACTIONS
        action_name     = ACTION_TYPES[action_type_idx]
        cost            = self.env._apply_action(action_name, target_idx)
        self._steps    += 1

        target_id = None
        if target_idx < len(self.env._top_risk_nodes):
            target_id = self.env._top_risk_nodes[target_idx]

        return {
            "step":        self._steps,
            "action":      action_name,
            "target":      target_id,
            "cost":        round(cost, 3),
            "description": self._describe(action_name, target_id),
        }

    def _describe(self, action: str, target: Optional[int]) -> str:
        zone_type = ""
        if target and target in self.env.twin.nodes:
            zone_type = self.env.twin.nodes[target].zone_type
        return {
            "do_nothing":     "Monitoring — no intervention needed",
            "open_drainage":  f"Opening drainage at Zone {target} ({zone_type})",
            "reroute_edge":   "Rerouting traffic away from congested corridor",
            "reduce_signal":  f"Green wave at Zone {target} ({zone_type})",
            "green_corridor": "Activating low-AQI green corridor",
            "emergency_drain":f"EMERGENCY drainage at Zone {target} ({zone_type})",
        }.get(action, action)


# ─────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    from core.twin_engine import TwinEngine
    from config import GRAPH_CACHE

    # Install correct versions if needed:
    # pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    # pip install stable-baselines3==2.1.0

    twin = TwinEngine(cache_path=GRAPH_CACHE)

    print("\n── Environment check ──")
    env = CityEnv(twin, max_steps=10)
    print(f"  Obs space   : {env.observation_space.shape}")
    print(f"  Obs dtype   : {env.observation_space.dtype}")
    print(f"  Action space: {env.action_space.n}")

    obs, _ = env.reset()
    print(f"  Obs shape   : {obs.shape}")
    print(f"  Obs dtype   : {obs.dtype}")
    print(f"  Obs range   : [{obs.min():.3f}, {obs.max():.3f}]")

    print("\n── Random policy rollout (10 steps) ──")
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"  Step {i+1:2d} | "
              f"action={ACTION_TYPES[action % N_ACTIONS]:16s} | "
              f"reward={reward:+.4f}")
    print(f"  Total reward: {total_reward:+.4f}")

    print(f"\n── Quick PPO training (4096 steps) ──")
    model = train_agent(twin, timesteps=4096)

    print("\n── Inference test ──")
    agent = SelfHealingAgent(model, env)
    twin.set_rainfall(50.0)
    for i in range(5):
        twin.step()
        env._update_targets()
        result = agent.act()
        print(f"  Step {i+1} | {result['action']:16s} | {result['description']}")

    save_agent(model, os.path.join(MODEL_DIR, "rl_agent"))
    print("\n[ RLAgent ] All checks passed.")