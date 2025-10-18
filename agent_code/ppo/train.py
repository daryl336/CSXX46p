# agent_code/ppo_agent/train.py
import os
from typing import List

# Try relative import of callbacks - works when package executed as module
try:
    from . import callbacks as ppo_callbacks
except Exception:
    # fallback for other import styles
    import callbacks as ppo_callbacks

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = "models/ppo_agent.pth"  # consistent save/load location


# --------------------------------------------------------------------------------
# Setup Training Phase
# --------------------------------------------------------------------------------
def setup_training(self):
    """
    Called once before training starts (not each round).
    Initializes logging and ensures train_agent exists. If not present,
    attempts several strategies to create/load it:
      1. reuse existing self.train_agent
      2. infer obs-dim from self.last_obs and build agent
      3. call callbacks.setup(self) (if it initializes agent)
      4. fallback to default input_dim (warn) and build agent
    Also loads existing model checkpoint if present.
    """
    self.logger.info("Setting up PPO training.")
    self.round_counter = getattr(self, "round_counter", 0)

    # Ensure model dir exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir != "":
        os.makedirs(model_dir, exist_ok=True)

    # 1) If already initialized by callbacks, reuse it
    if hasattr(self, "train_agent") and self.train_agent is not None:
        self.logger.info("Using existing train_agent (already initialized).")
    else:
        # 2) Try to infer input dim from last_obs (most reliable)
        if hasattr(self, "last_obs") and self.last_obs is not None:
            try:
                obs_vec = self.last_obs
                input_dim = int(len(obs_vec))
                action_dim = len(ACTIONS)
                self.train_agent = ppo_callbacks.PPOAgent(input_dim=input_dim, action_dim=action_dim)
                self.logger.info(f"Built PPOAgent from self.last_obs with input_dim={input_dim}.")
            except Exception as e:
                self.logger.warning(f"Failed to build PPOAgent from last_obs: {e}")
                self.train_agent = None

        # 3) Try calling callbacks.setup(self) if still missing (some repos init there)
        if (not hasattr(self, "train_agent")) or (self.train_agent is None):
            try:
                if hasattr(ppo_callbacks, "setup"):
                    self.logger.info("Attempting to call callbacks.setup(self) to initialize agent.")
                    try:
                        ppo_callbacks.setup(self)
                    except Exception as e:
                        # callbacks.setup may not create the agent (lazy init), so ignore errors
                        self.logger.debug(f"callbacks.setup(self) raised: {e}")
                # check again
                if hasattr(self, "train_agent") and self.train_agent is not None:
                    self.logger.info("train_agent initialized by callbacks.setup().")
            except Exception as e:
                self.logger.debug(f"callbacks.setup import/call failed: {e}")

        # 4) Fallback: create agent with default input_dim (warn)
        if (not hasattr(self, "train_agent")) or (self.train_agent is None):
            DEFAULT_INPUT_DIM = 50
            input_dim = DEFAULT_INPUT_DIM
            try:
                self.train_agent = ppo_callbacks.PPOAgent(input_dim=input_dim, action_dim=len(ACTIONS))
                self.logger.warning(
                    f"No existing agent/obs available. Built PPOAgent with fallback input_dim={input_dim}. "
                    "Feature extractor may produce different-sized vectors — consider making featurize deterministic."
                )
            except Exception as e:
                self.logger.error(f"Failed to construct fallback PPOAgent: {e}")
                raise RuntimeError("Unable to initialize PPO train_agent.") from e

    # --- Load existing model if compatible ---
    if os.path.exists(MODEL_PATH):
        try:
            self.train_agent.load(MODEL_PATH)
            self.logger.info(f"Loaded existing PPO model from '{MODEL_PATH}' for continued training.")
        except Exception as e:
            # Loading may fail due to mismatched input dims; warn and continue with fresh weights
            self.logger.warning(f"Failed to load existing PPO model ({e}). Continuing with current weights.")

    self.logger.info("setup_training completed.")


# --------------------------------------------------------------------------------
# Game Step Callback
# --------------------------------------------------------------------------------
def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called each time an action was performed.
    Logs transition (s, a, r, s') to PPO buffer.
    """
    if old_game_state is None or new_game_state is None:
        return

    # --- Reward shaping (customize as needed) ---
    reward = 0
    if "COIN_COLLECTED" in events:
        reward += 10
    if "KILLED_OPPONENT" in events:
        reward += 50
    if "KILLED_SELF" in events:
        reward -= 100
    if "INVALID_ACTION" in events:
        reward -= 5
    if "SURVIVED_ROUND" in events:
        reward += 20

    done = False

    # Store only if train_agent exists
    if hasattr(self, "train_agent") and self.train_agent is not None:
        try:
            self.train_agent.store_transition(
                obs=self.last_obs,
                action=self.last_action,
                log_prob=self.last_log_prob,
                reward=reward,
                value=self.last_value,
                done=done
            )
        except Exception as e:
            self.logger.warning(f"Failed to store transition: {e}")
    else:
        self.logger.debug("train_agent not present in game_events_occurred — skipping store.")


# --------------------------------------------------------------------------------
# End of Round Callback
# --------------------------------------------------------------------------------
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once at the end of each round.
    Finalizes reward, updates PPO, and saves model checkpoint.
    """
    final_reward = 0
    if "SURVIVED_ROUND" in events:
        final_reward += 30
    if "KILLED_SELF" in events:
        final_reward -= 50

    if hasattr(self, "train_agent") and self.train_agent is not None:
        try:
            # Add last transition as terminal
            self.train_agent.store_transition(
                obs=self.last_obs,
                action=self.last_action,
                log_prob=self.last_log_prob,
                reward=final_reward,
                value=self.last_value,
                done=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to store final transition: {e}")

        # Update and save
        try:
            self.train_agent.update()
        except Exception as e:
            self.logger.warning(f"PPO update failed: {e}")

        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            self.train_agent.save(MODEL_PATH)
            self.logger.info(f"Model checkpoint saved to '{MODEL_PATH}'")
        except Exception as e:
            self.logger.warning(f"Failed to save PPO model: {e}")
    else:
        self.logger.warning("train_agent missing at end_of_round — skipping update/save.")

    self.round_counter = getattr(self, "round_counter", 0) + 1
    self.logger.info(f"Round {self.round_counter} ended. (training hook complete)")