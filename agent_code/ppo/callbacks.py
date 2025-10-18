# callbacks.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MODEL_PATH = "ppo_agent.pth"  # save/load path


# ---------------------------------------------------------------------
# PPO Network
# ---------------------------------------------------------------------
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        # input_dim is dynamic now
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value


# ---------------------------------------------------------------------
# PPO Agent Core Class
# ---------------------------------------------------------------------
class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, update_epochs=4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPONetwork(input_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Experience buffer (cleared each round)
        self.memory = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

    def featurize(self, game_state: dict) -> np.ndarray:
        """
        Converts game state dict to numeric vector for the neural network.
        Replace/extend this with your actual feature extraction logic.
        """
        if game_state is None:
            return np.zeros(50, dtype=np.float32)

        # Example featurization (this is likely producing a longer vector in your case).
        # Keep this consistent with your expectations or update the network to match.
        try:
            board = np.array(game_state["field"]).flatten()
        except Exception:
            board = np.zeros(0, dtype=np.float32)

        # self position (x,y) if present
        try:
            position = np.array(game_state["self"][3], dtype=np.float32)
        except Exception:
            position = np.zeros(2, dtype=np.float32)

        # optional: bombs, coins, others — extend as required
        features = np.concatenate([board.astype(np.float32), position.astype(np.float32)])
        return features

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs, value = self.model(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # return python scalars
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(self, obs, action, log_prob, reward, value, done):
        self.memory["obs"].append(obs)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)

    def compute_advantages(self, next_value=0):
        rewards = np.array(self.memory["rewards"], dtype=np.float32)
        values = np.array(self.memory["values"] + [next_value], dtype=np.float32)
        dones = np.array(self.memory["dones"], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self):
        if len(self.memory["obs"]) == 0:
            return

        obs = torch.tensor(np.vstack(self.memory["obs"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32).to(self.device)

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        for _ in range(self.update_epochs):
            probs, values = self.model(obs)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * torch.tensor(advantages, dtype=torch.float32).to(self.device)
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * torch.tensor(advantages, dtype=torch.float32).to(self.device)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            loss = actor_loss + critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # clear memory
        for k in self.memory.keys():
            self.memory[k] = []

    def save(self, path=MODEL_PATH):
        torch.save(self.model.state_dict(), path)

    def load(self, path=MODEL_PATH):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))


# ---------------------------------------------------------------------
# Helper: build agent using observed feature length
# ---------------------------------------------------------------------
def _build_agent_with_obs(self, obs_vector):
    """
    Create a PPOAgent using the length of obs_vector.
    If a saved model exists, it will be loaded after construction.
    """
    input_dim = int(np.array(obs_vector).shape[0])
    action_dim = len(ACTIONS)
    agent = PPOAgent(input_dim=input_dim, action_dim=action_dim)
    # If a model checkpoint exists, try to load it
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            self.logger.info(f"Loaded pretrained PPO weights from {MODEL_PATH} for input_dim={input_dim}")
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained weights: {e} (continuing with random init)")
    else:
        self.logger.info(f"No pretrained model found. Initialized new PPOAgent with input_dim={input_dim}")
    return agent


# ---------------------------------------------------------------------
# Bomberman Callbacks — Required by the framework
# ---------------------------------------------------------------------
def setup(self):
    """
    Called once before the first game starts.
    We intentionally do NOT build the agent here because we need the
    real feature vector length produced by featurize(game_state).
    Instead just prepare paths / counters and optionally load a small config.
    """
    # placeholders for lazy init
    self.train_agent = None
    self.last_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None

    # optional: if you want to force a specific input_dim, create agent here:
    # input_dim = 50
    # self.train_agent = PPOAgent(input_dim, len(ACTIONS))

    self.logger.info("PPO callbacks.setup() finished. Agent will be lazily initialized at first act().")


def act(self, game_state):
    """
    Called each game step.
    We lazily initialize the PPOAgent on first call so that input_dim matches featurize().
    """
    # If agent uninitialized, create it using the featurized obs length
    # BUT we need a temporary featurizer — we'll use the method from PPOAgent class.
    # If train_agent is None, instantiate a temporary agent for featurize only.
    if self.train_agent is None:
        # temporary minimal agent to access featurize (without a model)
        temp_agent = PPOAgent(input_dim=1, action_dim=len(ACTIONS))
        obs_vec = temp_agent.featurize(game_state)
        obs_len = int(np.array(obs_vec).shape[0])
        # build the actual agent with proper input_dim and assign to self
        self.train_agent = _build_agent_with_obs(self, obs_vec)
        # overwrite the featurize method if you prefer a custom extractor; otherwise existing featurize used.

        # log the observed feature length to help debugging
        self.logger.info(f"PPO lazy-init: featurize produced obs length = {obs_len}. Agent built with input_dim={obs_len}")

    # Now normal action selection
    obs = self.train_agent.featurize(game_state)
    # Sanity check obs length
    if not hasattr(self.train_agent, "model"):
        raise RuntimeError("train_agent has no model attribute after initialization.")
    expected_in = list(self.train_agent.model.actor[0].weight.shape)[1]  # input_dim from first Linear
    if int(np.array(obs).shape[0]) != expected_in:
        # If mismatch still occurs (shouldn't), attempt to rebuild agent
        self.logger.warning(f"Feature length mismatch (obs {obs.shape[0]} vs model expects {expected_in}). Rebuilding agent.")
        self.train_agent = _build_agent_with_obs(self, obs)
        self.logger.info("Agent rebuilt to match observation dimension.")

    action_idx, log_prob, value = self.train_agent.select_action(obs)

    # Store for later use in training callbacks
    self.last_obs = obs
    self.last_action = action_idx
    self.last_log_prob = log_prob
    self.last_value = value

    return ACTIONS[action_idx]