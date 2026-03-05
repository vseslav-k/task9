import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Task8.DQNAgent import DQNAgent
from Task6.CartpoleDQN import DQNAgent as DQNAgentOld
from Task7.AsyncDRL import AsyncAgent
from Task7.AsyncDRL import QNet
from Task7.AsyncDRL import SharedAdam

import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

# ===== ADDED: TensorBoard setup + helpers =====
import time
from collections import deque

RUN_ID = time.strftime("%Y%m%d-%H%M%S")

AGENT_NAMES = ["AsyncAgent", "DQNNew", "DQNOld"]
writers = {name: SummaryWriter(log_dir=f"runs/{RUN_ID}/{name}") for name in AGENT_NAMES}
reward_windows = {name: deque(maxlen=100) for name in AGENT_NAMES}

def agent_name(i: int) -> str:
    return AGENT_NAMES[i] if 0 <= i < len(AGENT_NAMES) else f"agent_{i}"

def get_exploration(agent_obj) -> float:
    # support common names across your agents
    if hasattr(agent_obj, "exploration_rate"):
        try:
            return float(agent_obj.exploration_rate)
        except Exception:
            pass
    if hasattr(agent_obj, "epsilon"):
        try:
            return float(agent_obj.epsilon)
        except Exception:
            pass
    return float("nan")

def set_no_exploration(agent_obj):
    if hasattr(agent_obj, "exploration_rate"):
        try:
            agent_obj.exploration_rate = 0.0
        except Exception:
            pass
    if hasattr(agent_obj, "epsilon"):
        try:
            agent_obj.epsilon = 0.0
        except Exception:
            pass

def log_training_error(agent_obj, w: SummaryWriter, step: int):
    if not hasattr(agent_obj, "training_error"):
        return
    try:
        arr = agent_obj.training_error
        if arr is None or len(arr) == 0:
            return
        last = float(arr[-1])
        tail = arr[-100:] if len(arr) > 100 else arr
        mean100 = float(np.mean(tail))
        w.add_scalar("train/error_last", last, step)
        w.add_scalar("train/error_mean_100", mean100, step)
    except Exception:
        return

def decay(agent_obj):
    if hasattr(agent_obj, "decay_exploration_rate"):
        agent_obj.decay_exploration_rate()
    elif hasattr(agent_obj, "decay_epsilon"):
        agent_obj.decay_epsilon()
# ===== END ADDED =====


train_episodes = 100

env = gym.make("CartPole-v1")
learning_rate = 1e-3
init_exploration_rate = 1
exploration_rate_decay = init_exploration_rate / train_episodes
min_exploration_rate = 0.07
agentOld = DQNAgentOld(env, learning_rate, init_exploration_rate, exploration_rate_decay, min_exploration_rate)


env2 = gym.make("CartPole-v1")
    
learning_rate = 1e-3
init_exploration_rate = 1
exploration_rate_decay_exp = 0.995
min_exploration_rate = 0.03
agent = DQNAgent(env2, learning_rate, init_exploration_rate, exploration_rate_decay_exp, min_exploration_rate)
agent.train_sample = 128
agent.buffer_cap = 100000
agent.future_reward_discount_factor = 0.99
agent.q_target_update_rate = 500


env3 = gym.make("CartPole-v1")
lock = mp.Lock()

obs_dim = int(np.prod(env3.observation_space.shape))
action_dim = env3.action_space.n

online_nn = QNet(obs_dim, action_dim)
target_nn = QNet(obs_dim, action_dim)

adam = SharedAdam(online_nn.parameters(), lr=1e-3)

asyncAgent = AsyncAgent(
        env3,
        0,
        lock,
        init_exploration_rate,
        exploration_rate_decay,
        min_exploration_rate,
        online_nn,
        target_nn,
        adam,
        future_reward_discount_factor = 0.95,
        q_target_update_rate = 200,
    )

agents = [asyncAgent, agent, agentOld]


def train_agents(episodes: int = 500, max_steps: int = 500, seed: int = 42):
    for i in range(len(agents)):
        agents[i].env = gym.make("CartPole-v1")

    global_step = 0

    for ep in range(episodes):
        total_reward_for_agent = [0.0, 0.0, 0.0]
        ep_steps = 0
        for i in range(len(agents)):
            agent = agents[i]
            # ===== ADDED: per-agent deterministic seed so agents don't share identical resets =====
            obs, info = agent.env.reset(seed=seed + 1000 * i + ep)
            # ===== END ADDED =====
            for t in range(max_steps):

                action = agent.getAction(obs)
                next_obs, reward, terminated, truncated, info = agent.env.step(action)

                agent.update(obs, action, reward, next_obs, truncated or terminated)

                total_reward_for_agent[i] += float(reward)
                obs = next_obs
                ep_steps += 1
                global_step += 1

                if terminated or truncated:
                    break

            decay(agent)

            # ===== ADDED: TensorBoard logs (episode granularity) =====
            name = agent_name(i)
            w = writers[name]

            w.add_scalar("train/episode_return", float(total_reward_for_agent[i]), ep)
            reward_windows[name].append(float(total_reward_for_agent[i]))
            w.add_scalar("train/avg_return_100", float(np.mean(reward_windows[name])), ep)

            w.add_scalar("train/exploration_rate", get_exploration(agent), ep)

            log_training_error(agent, w, ep)
            # ===== END ADDED =====

        # ===== ADDED: convenience comparison series across runs =====
        try:
            writers["AsyncAgent"].add_scalar("compare/return_async", float(total_reward_for_agent[0]), ep)
            writers["DQNNew"].add_scalar("compare/return_dqn_new", float(total_reward_for_agent[1]), ep)
            writers["DQNOld"].add_scalar("compare/return_dqn_old", float(total_reward_for_agent[2]), ep)
        except Exception:
            pass
        # ===== END ADDED =====

    for a in agents:
        a.env.close()


def test_agents(episodes: int = 500, max_steps: int = 500, seed: int = 42):
    for i in range(len(agents)):
        agents[i].env = gym.make("CartPole-v1", render_mode = "human")
        # ===== CHANGED (minimal): use helper so it works for epsilon/exploration_rate =====
        set_no_exploration(agents[i])
        # ===== END CHANGED =====

    global_step = 0

    for ep in range(episodes):
        total_reward_for_agent = [0.0, 0.0, 0.0]
        ep_steps = 0
        for i in range(len(agents)):
            agent = agents[i]
            # ===== ADDED: per-agent deterministic seed =====
            obs, info = agent.env.reset(seed=seed + 1000 * i + ep)
            # ===== END ADDED =====
            for t in range(max_steps):

                action = agent.getAction(obs)
                next_obs, reward, terminated, truncated, info = agent.env.step(action)

                total_reward_for_agent[i] += float(reward)
                obs = next_obs
                ep_steps += 1
                global_step += 1

                if terminated or truncated:
                    break

            # ===== ADDED: TensorBoard test logs =====
            name = agent_name(i)
            w = writers[name]
            w.add_scalar("test/episode_return", float(total_reward_for_agent[i]), ep)
            # ===== END ADDED =====

    for i in range(len(agents)):
        agents[i].env.close()

def main():
    train_agents(episodes=50)
    test_agents(episodes=50)

    # ===== ADDED: flush/close writers =====
    for w in writers.values():
        try:
            w.flush()
            w.close()
        except Exception:
            pass
    # ===== END ADDED =====


if(__name__ == "__main__"):
    main()