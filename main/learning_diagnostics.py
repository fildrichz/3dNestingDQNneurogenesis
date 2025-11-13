"""
Diagnostic script to check if DQN is actually learning.

This will help identify fundamental issues with learning:
1. Are Q-values changing over training?
2. Are losses decreasing?
3. What are reward magnitudes?
4. Is epsilon-greedy actually exploiting?
5. Are network parameters updating?
"""

import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

class LearningDiagnostics:
    """Track diagnostic metrics during training."""

    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = []
        self.q_value_means = []
        self.q_value_stds = []
        self.q_value_maxs = []
        self.q_value_mins = []
        self.epsilon_values = []
        self.reward_distributions = []
        self.action_entropies = []
        self.grad_norms = []

        # Per-episode tracking
        self.current_episode_rewards = []
        self.current_episode_q_values = []

    def on_step(self, reward, q_values, epsilon, loss=None, grad_norm=None):
        """Call this every step during training."""
        self.current_episode_rewards.append(reward)

        if q_values is not None and len(q_values) > 0:
            self.current_episode_q_values.append(q_values)

        if loss is not None:
            self.losses.append(loss)

        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

    def on_episode_end(self, epsilon):
        """Call this at end of episode."""
        if len(self.current_episode_rewards) > 0:
            ep_return = sum(self.current_episode_rewards)
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(len(self.current_episode_rewards))

            # Reward distribution
            self.reward_distributions.append({
                'mean': np.mean(self.current_episode_rewards),
                'std': np.std(self.current_episode_rewards),
                'min': np.min(self.current_episode_rewards),
                'max': np.max(self.current_episode_rewards),
            })

        if len(self.current_episode_q_values) > 0:
            all_q = np.concatenate(self.current_episode_q_values)
            self.q_value_means.append(np.mean(all_q))
            self.q_value_stds.append(np.std(all_q))
            self.q_value_maxs.append(np.max(all_q))
            self.q_value_mins.append(np.min(all_q))

        self.epsilon_values.append(epsilon)

        # Reset
        self.current_episode_rewards = []
        self.current_episode_q_values = []

    def print_summary(self, episode, window=50):
        """Print diagnostic summary."""
        if episode < 1:
            return

        print(f"\n{'='*80}")
        print(f"LEARNING DIAGNOSTICS - Episode {episode}")
        print(f"{'='*80}")

        # Returns
        if len(self.episode_returns) >= window:
            recent_returns = self.episode_returns[-window:]
            print(f"\nReturns (last {window} episodes):")
            print(f"  Mean: {np.mean(recent_returns):.3f} ± {np.std(recent_returns):.3f}")
            print(f"  Min: {np.min(recent_returns):.3f}")
            print(f"  Max: {np.max(recent_returns):.3f}")

            # Check if returns are improving
            if len(self.episode_returns) >= 2 * window:
                older_returns = self.episode_returns[-2*window:-window]
                improvement = np.mean(recent_returns) - np.mean(older_returns)
                print(f"  Improvement from previous {window}: {improvement:+.3f}")

        # Q-values
        if len(self.q_value_means) >= window:
            recent_q_means = self.q_value_means[-window:]
            recent_q_stds = self.q_value_stds[-window:]
            recent_q_maxs = self.q_value_maxs[-window:]
            recent_q_mins = self.q_value_mins[-window:]

            print(f"\nQ-values (last {window} episodes):")
            print(f"  Mean: {np.mean(recent_q_means):.3f} ± {np.std(recent_q_means):.3f}")
            print(f"  Range: [{np.mean(recent_q_mins):.3f}, {np.mean(recent_q_maxs):.3f}]")

            # Check if Q-values are changing
            if len(self.q_value_means) >= 2 * window:
                older_q_means = self.q_value_means[-2*window:-window]
                q_change = np.mean(recent_q_means) - np.mean(older_q_means)
                print(f"  Change from previous {window}: {q_change:+.3f}")

                if abs(q_change) < 0.01:
                    print(f"  ⚠️  WARNING: Q-values barely changing! Network may not be learning.")

        # Losses
        if len(self.losses) >= 100:
            recent_losses = self.losses[-100:]
            print(f"\nLosses (last 100 steps):")
            print(f"  Mean: {np.mean(recent_losses):.4f} ± {np.std(recent_losses):.4f}")
            print(f"  Min: {np.min(recent_losses):.4f}")
            print(f"  Max: {np.max(recent_losses):.4f}")

            if len(self.losses) >= 200:
                older_losses = self.losses[-200:-100]
                loss_change = np.mean(recent_losses) - np.mean(older_losses)
                print(f"  Change from previous 100: {loss_change:+.4f}")

                if loss_change > 0:
                    print(f"  ⚠️  WARNING: Loss increasing! May indicate instability.")

        # Rewards
        if len(self.reward_distributions) >= window:
            recent_rewards = self.reward_distributions[-window:]
            mean_rewards = [r['mean'] for r in recent_rewards]
            print(f"\nRewards per step (last {window} episodes):")
            print(f"  Mean: {np.mean(mean_rewards):.4f}")
            print(f"  Typical range: [{np.mean([r['min'] for r in recent_rewards]):.4f}, "
                  f"{np.mean([r['max'] for r in recent_rewards]):.4f}]")

        # Epsilon
        if len(self.epsilon_values) > 0:
            current_eps = self.epsilon_values[-1]
            print(f"\nExploration:")
            print(f"  Current epsilon: {current_eps:.3f}")
            print(f"  Exploitation rate: {(1-current_eps)*100:.1f}%")

            if current_eps > 0.5:
                print(f"  ⚠️  WARNING: Still exploring >50%. Network rarely exploits learned policy.")

        # Gradient norms
        if len(self.grad_norms) >= 100:
            recent_grads = self.grad_norms[-100:]
            print(f"\nGradient norms (last 100 steps):")
            print(f"  Mean: {np.mean(recent_grads):.4f}")
            print(f"  Max: {np.max(recent_grads):.4f}")

            if np.mean(recent_grads) < 1e-6:
                print(f"  ⚠️  WARNING: Gradients very small! Network may not be learning.")
            elif np.max(recent_grads) > 10:
                print(f"  ⚠️  WARNING: Large gradients detected! May indicate instability.")

        print(f"{'='*80}\n")

    def plot_diagnostics(self, save_path=None):
        """Plot diagnostic charts."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Returns
        if len(self.episode_returns) > 0:
            axes[0, 0].plot(self.episode_returns, alpha=0.3, label='Episode')
            if len(self.episode_returns) >= 10:
                window = min(50, len(self.episode_returns) // 4)
                moving_avg = np.convolve(self.episode_returns, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_returns)), moving_avg,
                               label=f'MA({window})', linewidth=2)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].set_title('Episode Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Q-values
        if len(self.q_value_means) > 0:
            axes[0, 1].plot(self.q_value_means, label='Mean', linewidth=2)
            axes[0, 1].fill_between(range(len(self.q_value_means)),
                                   self.q_value_mins, self.q_value_maxs,
                                   alpha=0.2, label='Range')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Q-value')
            axes[0, 1].set_title('Q-value Statistics')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Losses
        if len(self.losses) > 0:
            axes[0, 2].plot(self.losses, alpha=0.2)
            if len(self.losses) >= 100:
                window = 100
                moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                axes[0, 2].plot(range(window-1, len(self.losses)), moving_avg,
                               label=f'MA({window})', linewidth=2, color='red')
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)

        # Epsilon
        if len(self.epsilon_values) > 0:
            axes[1, 0].plot(self.epsilon_values, linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Exploration Rate')
            axes[1, 0].grid(True, alpha=0.3)

        # Reward distribution
        if len(self.reward_distributions) > 0:
            means = [r['mean'] for r in self.reward_distributions]
            stds = [r['std'] for r in self.reward_distributions]
            axes[1, 1].plot(means, label='Mean', linewidth=2)
            axes[1, 1].fill_between(range(len(means)),
                                   np.array(means) - np.array(stds),
                                   np.array(means) + np.array(stds),
                                   alpha=0.2, label='±1 std')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward per Step')
            axes[1, 1].set_title('Step Rewards')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Gradient norms
        if len(self.grad_norms) > 0:
            axes[1, 2].plot(self.grad_norms, alpha=0.2)
            if len(self.grad_norms) >= 100:
                window = 100
                moving_avg = np.convolve(self.grad_norms, np.ones(window)/window, mode='valid')
                axes[1, 2].plot(range(window-1, len(self.grad_norms)), moving_avg,
                               label=f'MA({window})', linewidth=2, color='red')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Gradient Norm')
            axes[1, 2].set_title('Gradient Magnitude')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diagnostic plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def add_diagnostics_to_training(agent, diagnostics):
    """Monkey-patch agent to track gradient norms."""
    original_train_step = agent.train_step

    def instrumented_train_step():
        loss = original_train_step()

        if loss is not None:
            # Compute gradient norm
            total_norm = 0.0
            for p in agent.q.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            diagnostics.grad_norms.append(total_norm)

        return loss

    agent.train_step = instrumented_train_step
    return agent
