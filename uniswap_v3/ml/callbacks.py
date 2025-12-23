"""
Custom Callbacks for PPO Training

Includes reward logging and multi-pool training callbacks.
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


class RewardLoggingCallback(BaseCallback):
    """
    Log episode rewards and metrics to TensorBoard.

    Tracks:
    - Episode reward
    - Cumulative fees, IL, gas costs
    - Number of rebalances
    - Net return
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []

    def _on_step(self) -> bool:
        """Called after each environment step"""

        # Check if episode finished
        if self.locals.get('dones') is not None and self.locals.get('dones')[0]:
            info = self.locals.get('infos', [{}])[0]

            # Extract episode info
            episode_info = info.get('episode', {})
            episode_reward = episode_info.get('r', 0)
            episode_length = episode_info.get('l', 0)

            # Log to TensorBoard
            if self.logger is not None:
                self.logger.record("episode/reward", episode_reward)
                self.logger.record("episode/length", episode_length)
                self.logger.record("episode/cumulative_fees",
                                 info.get('cumulative_fees', 0))
                self.logger.record("episode/cumulative_il",
                                 info.get('cumulative_il', 0))
                self.logger.record("episode/cumulative_gas",
                                 info.get('cumulative_gas', 0))
                self.logger.record("episode/total_rebalances",
                                 info.get('total_rebalances', 0))
                self.logger.record("episode/net_return",
                                 info.get('net_return', 0))

                # Calculate and log APR if episode completed
                if episode_length > 0:
                    # Assume 720 hours per episode (30 days)
                    days = episode_length / 24
                    if days > 0 and 'net_return' in info:
                        # Annualized return
                        apr = (info['net_return'] / 10000) * (365 / days)
                        self.logger.record("episode/apr", apr)

            # Store for later analysis
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_metrics.append({
                'fees': info.get('cumulative_fees', 0),
                'il': info.get('cumulative_il', 0),
                'gas': info.get('cumulative_gas', 0),
                'rebalances': info.get('total_rebalances', 0),
                'net_return': info.get('net_return', 0),
                'length': episode_length
            })

            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                print(f"\n[Episode {len(self.episode_rewards)}] "
                      f"Reward: {episode_reward:.2f}, "
                      f"Net: ${info.get('net_return', 0):.2f}, "
                      f"Rebalances: {info.get('total_rebalances', 0)}")

        return True


class MultiPoolCallback(BaseCallback):
    """
    Switch between different pools during training.

    This ensures the agent learns from diverse market conditions across
    multiple pools and protocols.
    """

    def __init__(self, pool_data_files: List[Path], episodes_per_pool: int = 5,
                 verbose=0):
        """
        Initialize multi-pool callback.

        Args:
            pool_data_files: List of paths to pool data parquet files
            episodes_per_pool: Number of episodes to train on each pool before switching
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.pool_data_files = pool_data_files
        self.episodes_per_pool = episodes_per_pool
        self.current_pool_idx = 0
        self.episodes_on_current_pool = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        """Called after each environment step"""

        # Check if episode finished
        if self.locals.get('dones') is not None and self.locals.get('dones')[0]:
            self.episodes_on_current_pool += 1
            self.total_episodes += 1

            # Switch pool after N episodes
            if self.episodes_on_current_pool >= self.episodes_per_pool:
                # Move to next pool (cycle through)
                self.current_pool_idx = (self.current_pool_idx + 1) % len(self.pool_data_files)
                self.episodes_on_current_pool = 0

                # Load new pool data
                try:
                    new_pool_file = self.pool_data_files[self.current_pool_idx]
                    new_pool_data = pd.read_parquet(new_pool_file)

                    # Update environment's historical data
                    # Note: This requires the environment to have an update method
                    if hasattr(self.training_env, 'env_method'):
                        self.training_env.env_method('update_historical_data', new_pool_data)

                    if self.verbose > 0:
                        pool_name = new_pool_file.stem
                        print(f"\n{'='*60}")
                        print(f"[Episode {self.total_episodes}] Switched to pool "
                              f"{self.current_pool_idx + 1}/{len(self.pool_data_files)}")
                        print(f"Pool: {pool_name}")
                        print(f"{'='*60}\n")

                    # Log pool switch to TensorBoard
                    if self.logger is not None:
                        self.logger.record("train/current_pool_idx", self.current_pool_idx)

                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Failed to switch pool: {e}")

        return True


class ProgressCallback(BaseCallback):
    """
    Display training progress periodically.

    Shows:
    - Current timestep / total
    - Episodes completed
    - Estimated time remaining
    """

    def __init__(self, total_timesteps: int, check_freq: int = 10000, verbose=1):
        """
        Initialize progress callback.

        Args:
            total_timesteps: Total training timesteps
            check_freq: How often to print progress (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.start_time = None
        self.last_check_time = None

    def _on_training_start(self):
        """Called at training start"""
        import time
        self.start_time = time.time()
        self.last_check_time = time.time()

    def _on_step(self) -> bool:
        """Called after each environment step"""

        if self.n_calls % self.check_freq == 0 and self.verbose > 0:
            import time
            current_time = time.time()

            # Calculate progress
            progress = (self.num_timesteps / self.total_timesteps) * 100

            # Estimate time remaining
            elapsed = current_time - self.start_time
            if self.num_timesteps > 0:
                steps_per_sec = self.num_timesteps / elapsed
                remaining_steps = self.total_timesteps - self.num_timesteps
                eta_seconds = remaining_steps / steps_per_sec
                eta_hours = eta_seconds / 3600

                print(f"\n[Progress] {self.num_timesteps:,}/{self.total_timesteps:,} steps "
                      f"({progress:.1f}%) | "
                      f"Speed: {steps_per_sec:.0f} steps/s | "
                      f"ETA: {eta_hours:.1f}h")

            self.last_check_time = current_time

        return True
