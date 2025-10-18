# training_manager.py
"""
TrainingManager for Bomberman PPO training (non-gym hooks).
- Provides curriculum stage management (progressive difficulty).
- Provides evaluation scheduling (mark next N rounds as evaluation).
- TensorBoard logging (training + evaluation metrics).
- Checkpoint management (latest + best).
Usage:
  - Create TrainingManager(...) instance in callbacks.setup or train.setup_training.
  - Call manager.setup(self) from train.setup_training (or callbacks.setup) once agent exists.
  - Call manager.on_round_end(self, round_metrics) from train.end_of_round.
  - To start an evaluation block (next N rounds are evaluation-only), call manager.start_evaluation(n).
Notes:
  - This manager expects your train hooks to pass per-round metric dicts to on_round_end.
  - It does not manage environment changes (map complexity) automatically; it exposes `current_stage`
    so you can read it in your env factory to adjust map difficulty.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    # TensorBoard optional, fallback to no-op writer
    SummaryWriter = None


@dataclass
class StageSpec:
    name: str
    rounds: int  # number of rounds to train in this stage
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingManager:
    def __init__(
        self,
        *,
        model_path: str = "models/ppo_agent.pth",
        best_model_path: str = "models/ppo_agent_best.pth",
        log_dir: str = "logs/ppo",
        stages: Optional[List[StageSpec]] = None,
        eval_every_rounds: int = 50,
        eval_episodes: int = 20,
        save_every_rounds: int = 5,
        best_metric_name: str = "eval/mean_reward",  # key in eval metrics to compare for best model
        minimize_metric: bool = False,  # False -> higher is better
    ):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.model_path = model_path
        self.best_model_path = best_model_path
        self.log_dir = log_dir
        self.save_every_rounds = max(1, int(save_every_rounds))
        self.eval_every_rounds = max(1, int(eval_every_rounds))
        self.eval_episodes_cfg = max(1, int(eval_episodes))
        self.best_metric_name = best_metric_name
        self.minimize_metric = bool(minimize_metric)

        # default curriculum stages if none provided
        if stages is None:
            self.stages = [
                StageSpec("stage_1_empty", rounds=50, description="Empty map, coins only"),
                StageSpec("stage_2_crates", rounds=100, description="Add crates; coin gathering"),
                StageSpec("stage_3_bombs", rounds=150, description="Bombs and safety learning"),
                StageSpec("stage_4_bombing", rounds=200, description="Bomb placement & crate destruction"),
                StageSpec("stage_5_simple_enemies", rounds=300, description="Add simple random enemies"),
                StageSpec("stage_6_full", rounds=9999, description="Full complexity"),
            ]
        else:
            self.stages = stages

        # internal counters/state
        self.total_rounds_trained = 0  # overall rounds across stages
        self.stage_idx = 0
        self.stage_round_counter = 0  # rounds completed in current stage
        self.last_checkpoint_time = 0.0

        # evaluation state
        self.in_evaluation_mode = False
        self.eval_target_episodes = 0
        self.eval_collected = 0
        self.eval_episode_metrics: List[Dict[str, Any]] = []

        # best metric tracking
        self.best_metric_value = None
        self.writer = None

        # external reference to the agent object (set in setup)
        self.agent_self = None

        # bookkeeping files
        self._meta_file = os.path.join(self.log_dir, "training_manager_meta.json")

    # -------------------------
    # Setup & persistence
    # -------------------------
    def setup(self, agent_self):
        """
        Call this once after your agent is initialized (train_agent exists).
        - agent_self: the agent object (self from callbacks/train) so we can call load/save on agent.train_agent
        """
        self.agent_self = agent_self
        # Initialize TensorBoard summary writer if available
        try:
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=self.log_dir)
                self._log_info("TensorBoard SummaryWriter created at %s" % self.log_dir)
            else:
                self.writer = None
                self._log_info("torch.utils.tensorboard not available; skipping TB writer.")
        except Exception as e:
            self.writer = None
            self._log_info(f"Failed to create TensorBoard writer: {e}")

        # Try to load meta and best metric
        if os.path.exists(self._meta_file):
            try:
                with open(self._meta_file, "r") as f:
                    meta = json.load(f)
                    self.total_rounds_trained = meta.get("total_rounds_trained", 0)
                    self.stage_idx = meta.get("stage_idx", 0)
                    self.stage_round_counter = meta.get("stage_round_counter", 0)
                    self.best_metric_value = meta.get("best_metric_value", None)
                    self._log_info("Loaded training_manager meta from file.")
            except Exception as e:
                self._log_info(f"Failed to load meta file: {e}")

        # Attempt to load agent checkpoint (if compatible)
        if getattr(agent_self, "train_agent", None) is not None:
            try:
                if os.path.exists(self.model_path):
                    agent_self.train_agent.load(self.model_path)
                    self._log_info(f"Loaded checkpoint from {self.model_path}")
                else:
                    self._log_info("No checkpoint found to load.")
            except Exception as e:
                self._log_info(f"Failed to load checkpoint: {e}")

        # record start timestamp
        self.last_checkpoint_time = time.time()

    def _save_meta(self):
        try:
            with open(self._meta_file, "w") as f:
                json.dump({
                    "total_rounds_trained": self.total_rounds_trained,
                    "stage_idx": self.stage_idx,
                    "stage_round_counter": self.stage_round_counter,
                    "best_metric_value": self.best_metric_value
                }, f)
        except Exception as e:
            self._log_info(f"Failed to save meta file: {e}")

    # -------------------------
    # Utilities / logging
    # -------------------------
    def _log_info(self, msg: str):
        # safe logger: prefer agent_self.logger if present
        try:
            if self.agent_self is not None and hasattr(self.agent_self, "logger"):
                self.agent_self.logger.info(f"[TrainingManager] {msg}")
            else:
                print(f"[TrainingManager] {msg}")
        except Exception:
            print(f"[TrainingManager] {msg}")

    def _tb_log(self, key: str, value: float, step: int):
        if self.writer is None:
            return
        try:
            self.writer.add_scalar(key, float(value), step)
        except Exception:
            pass

    # -------------------------
    # Stage and Curriculum API
    # -------------------------
    def current_stage(self) -> StageSpec:
        return self.stages[min(self.stage_idx, len(self.stages)-1)]

    def advance_stage_if_needed(self):
        spec = self.current_stage()
        if self.stage_round_counter >= spec.rounds:
            # advance
            if self.stage_idx < len(self.stages) - 1:
                self.stage_idx += 1
                self.stage_round_counter = 0
                self._log_info(f"Advancing to stage {self.stage_idx}: {self.current_stage().name}")
                # optionally save a stage checkpoint
                self.save_checkpoint(suffix=f"stage_{self.stage_idx}")
                self._save_meta()
            else:
                # last stage has very large rounds, do not advance
                pass

    # -------------------------
    # Evaluation / training toggles
    # -------------------------
    def start_evaluation(self, n_episodes: Optional[int] = None):
        """
        Mark the next n_episodes rounds as evaluation-only. While in eval mode, training updates
        should be suppressed by your code (i.e. skip PPO update when manager.in_evaluation_mode==True)
        The manager will collect eval metrics from on_round_end until n_episodes are collected, then compute aggregated metrics.
        """
        self.in_evaluation_mode = True
        self.eval_target_episodes = n_episodes or self.eval_episodes_cfg
        self.eval_collected = 0
        self.eval_episode_metrics = []
        self._log_info(f"Evaluation started for next {self.eval_target_episodes} episodes.")

    def _finish_evaluation(self):
        # compute aggregated metrics
        if len(self.eval_episode_metrics) == 0:
            self._log_info("No evaluation episodes collected.")
            self.in_evaluation_mode = False
            return None

        # Aggregation: compute mean for numeric metrics
        agg = {}
        keys = set().union(*(m.keys() for m in self.eval_episode_metrics))
        for k in keys:
            vals = [m[k] for m in self.eval_episode_metrics if isinstance(m.get(k, None), (int, float))]
            if len(vals) > 0:
                agg[k] = float(sum(vals) / len(vals))
        agg["n_episodes"] = len(self.eval_episode_metrics)

        # Log to TB
        step = self.total_rounds_trained
        for k, v in agg.items():
            self._tb_log(f"eval/{k}", v, step)

        # Best model logic
        best_metric = agg.get(self.best_metric_name.split("/", 1)[-1], None)
        if best_metric is not None:
            better = (self.best_metric_value is None) or \
                     ((best_metric < self.best_metric_value) if self.minimize_metric else (best_metric > self.best_metric_value))
            if better:
                # new best
                self.best_metric_value = best_metric
                try:
                    if getattr(self.agent_self, "train_agent", None) is not None:
                        self.agent_self.train_agent.save(self.best_model_path)
                        self._log_info(f"New best model saved to {self.best_model_path} (metric {self.best_metric_name}={best_metric:.4f})")
                except Exception as e:
                    self._log_info(f"Failed to save best model: {e}")

        # Clear eval mode and save summary
        self.in_evaluation_mode = False
        self.eval_target_episodes = 0
        self.eval_collected = 0
        self.eval_episode_metrics = []
        self._save_meta()
        self._log_info(f"Evaluation finished. Aggregated metrics: {agg}")
        return agg

    # -------------------------
    # Round-end hook (call from your end_of_round)
    # -------------------------
    def on_round_end(self, round_metrics: Dict[str, Any]):
        """
        Call this from train.end_of_round, with a metrics dict summarizing the round:
          e.g. {"total_reward": 12.5, "won": 1, "survival_time": 237, "coins": 3, "bombs_used": 4, ...}
        Manager will:
         - log training metrics to tensorboard
         - update counters, stage progression
         - collect eval episodes if in eval mode, and run finish when target reached
         - save checkpoint periodically
        """
        # bookkeeping counters
        self.total_rounds_trained += 1
        self.stage_round_counter += 1

        # Step index for TB
        step = self.total_rounds_trained

        # Distinguish eval vs train
        if self.in_evaluation_mode:
            self.eval_episode_metrics.append(round_metrics)
            self.eval_collected += 1
            self._log_info(f"Collected eval episode {self.eval_collected}/{self.eval_target_episodes}")
            if self.eval_collected >= self.eval_target_episodes:
                self._finish_evaluation()
            # Do NOT do training-specific saves/progression during eval unless you want to
            return

        # Training-mode logging: log provided metrics (flatten)
        for k, v in list(round_metrics.items()):
            if isinstance(v, (int, float)):
                self._tb_log(f"train/{k}", float(v), step)

        # Periodic checkpointing
        if (self.total_rounds_trained % self.save_every_rounds) == 0:
            self.save_checkpoint()

        # Periodic evaluation trigger (optional)
        if (self.total_rounds_trained % self.eval_every_rounds) == 0:
            # Start a short eval block automatically
            self.start_evaluation(self.eval_episodes_cfg)

        # Stage progression
        self.advance_stage_if_needed()
        self._save_meta()

    # -------------------------
    # Checkpointing
    # -------------------------
    def save_checkpoint(self, suffix: Optional[str] = None):
        if getattr(self.agent_self, "train_agent", None) is None:
            self._log_info("No agent present — skipping checkpoint.")
            return

        path = self.model_path
        if suffix:
            base, ext = os.path.splitext(self.model_path)
            path = f"{base}_{suffix}{ext}"

        try:
            self.agent_self.train_agent.save(path)
            self._log_info(f"Saved checkpoint to {path}")
            self._save_meta()
        except Exception as e:
            self._log_info(f"Failed to save checkpoint: {e}")

    # -------------------------
    # Close / cleanup
    # -------------------------
    def close(self):
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass