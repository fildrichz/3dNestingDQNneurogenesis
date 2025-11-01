"""
DQN Agent for Q4RealBPP 3D Bin Packing
Integrates DQN with Q4RealBPP-constrained packing environment
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from dqn_core.dqn import DQNAgent, DQNConfig
from nesting.packing_core_q4real import Q4RealBPPContainer, Box3D
from nesting.q4realbpp_loader import Item, Q4RealBPPInstance, Q4RealBPPLoader


@dataclass
class PackingState:
    """State representation for RL"""
    container: Q4RealBPPContainer
    remaining_items: List[Item]
    placed_items: List[Item]
    
    def is_terminal(self) -> bool:
        return len(self.remaining_items) == 0


class Q4RealBPPEnvironment:
    """
    RL Environment for Q4RealBPP 3D Bin Packing with constraints.
    Handles weight, affinity, load bearing, and load balancing constraints.
    """
    
    def __init__(self, instance: Q4RealBPPInstance, bin_id: int = 0):
        self.instance = instance
        self.bin_spec = instance.bins[bin_id]
        self.all_items = instance.items.copy()
        
        # State
        self.container: Optional[Q4RealBPPContainer] = None
        self.remaining_items: List[Item] = []
        self.placed_items: List[Item] = []
        self.current_item: Optional[Item] = None
        
        # For tracking
        self.step_count = 0
        self.total_reward = 0.0
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset environment.
        Returns: (obs, action_feats, action_mask)
        """
        # Create fresh container
        self.container = Q4RealBPPContainer(
            int(self.bin_spec.width),
            int(self.bin_spec.depth),
            int(self.bin_spec.height),
            self.bin_spec.max_weight,
            self.bin_spec.id
        )
        self.container.set_constraints(
            self.instance.affinities,
            self.instance.load_bearing_ratio,
            self.instance.load_balancing
        )
        
        # Reset items (optionally shuffle for variety)
        self.remaining_items = self.all_items.copy()
        np.random.shuffle(self.remaining_items)
        self.placed_items = []
        
        # Pick first item
        self.current_item = self.remaining_items[0]
        
        self.step_count = 0
        self.total_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action (place current item at EP with orientation).
        Returns: (next_obs, reward, done, info)
        """
        if self.current_item is None:
            raise ValueError("No current item - did you call reset()?")
        
        # Decode action
        ep, orientation = self._decode_action(action_idx)
        
        # Try placement
        success = False
        if ep is not None and orientation is not None:
            w, d, h = orientation
            success = self.container.place_at(
                ep, (w, d, h),
                self.current_item.weight,
                self.current_item.category,
                self.current_item.id
            )
        
        # Calculate reward
        reward = self._calculate_reward(success)
        self.total_reward += reward
        
        # Update state
        if success:
            self.placed_items.append(self.current_item)
            self.remaining_items.remove(self.current_item)
        else:
            # Item couldn't be placed - remove from queue anyway
            self.remaining_items.remove(self.current_item)
        
        # Check if done
        done = len(self.remaining_items) == 0
        
        # Move to next item
        if not done:
            self.current_item = self.remaining_items[0]
        else:
            self.current_item = None
        
        self.step_count += 1
        
        # Info
        info = {
            'success': success,
            'items_placed': len(self.placed_items),
            'items_remaining': len(self.remaining_items),
            'weight_used': self.container.current_weight,
            'weight_capacity': self.container.max_weight,
            'volume_utilization': self._get_volume_utilization(),
        }
        
        # Get next observation
        if done:
            # Return dummy observation
            obs = np.zeros(self._get_obs_dim())
            action_feats = np.zeros((self._get_max_actions(), self._get_action_feat_dim()))
            mask = np.zeros(self._get_max_actions())
        else:
            obs, action_feats, mask = self._get_observation()
        
        return (obs, action_feats, mask), reward, done, info
    
    def _decode_action(self, action_idx: int) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Decode action index to (EP, orientation)"""
        eps = sorted(set(self.container.eps), key=lambda p: (p[2], p[1], p[0]))
        orientations = [
            (int(self.current_item.width), int(self.current_item.depth), int(self.current_item.height)),
            (int(self.current_item.width), int(self.current_item.height), int(self.current_item.depth)),
            (int(self.current_item.depth), int(self.current_item.width), int(self.current_item.height)),
            (int(self.current_item.depth), int(self.current_item.height), int(self.current_item.width)),
            (int(self.current_item.height), int(self.current_item.width), int(self.current_item.depth)),
            (int(self.current_item.height), int(self.current_item.depth), int(self.current_item.width)),
        ]
        
        num_eps = len(eps)
        num_orients = len(orientations)
        
        if action_idx >= num_eps * num_orients:
            return None, None
        
        ep_idx = action_idx // num_orients
        orient_idx = action_idx % num_orients
        
        return eps[ep_idx], orientations[orient_idx]
    
    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current observation.
        Returns: (state_vector, action_features, action_mask)
        """
        # State vector: container state + current item features
        obs = self._encode_state()
        
        # Action features: EP + orientation features for each possible action
        action_feats, action_mask = self._encode_actions()
        
        return obs, action_feats, action_mask
    
    def _encode_state(self) -> np.ndarray:
        """Encode container and item state"""
        # Container features
        W, D, H = self.container.w, self.container.d, self.container.h
        volume_util = self._get_volume_utilization()
        weight_util = self.container.current_weight / self.container.max_weight
        num_placed = len(self.container.placed)
        num_remaining = len(self.remaining_items)
        
        # Current item features (normalized)
        if self.current_item:
            item_w = self.current_item.width / W
            item_d = self.current_item.depth / D
            item_h = self.current_item.height / H
            item_weight = self.current_item.weight / self.container.max_weight
            item_cat = self.current_item.category / 10.0  # Assume max 10 categories
        else:
            item_w = item_d = item_h = item_weight = item_cat = 0.0
        
        # Categories in bin (binary encoding - simple approach)
        cats_in_bin = [1.0 if i in self.container.categories_in_bin else 0.0 
                       for i in range(10)]  # Assume max 10 categories
        
        state = np.array([
            volume_util,
            weight_util,
            num_placed / 100.0,  # Normalize
            num_remaining / 100.0,
            item_w,
            item_d,
            item_h,
            item_weight,
            item_cat,
            *cats_in_bin,
        ], dtype=np.float32)
        
        return state
    
    def _encode_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode possible actions (EP + orientation combinations).
        Returns: (action_features, action_mask)
        """
        if not self.current_item:
            # No current item
            max_actions = self._get_max_actions()
            action_feats = np.zeros((max_actions, self._get_action_feat_dim()), dtype=np.float32)
            mask = np.zeros(max_actions, dtype=np.float32)
            return action_feats, mask
        
        eps = sorted(set(self.container.eps), key=lambda p: (p[2], p[1], p[0]))
        orientations = [
            (int(self.current_item.width), int(self.current_item.depth), int(self.current_item.height)),
            (int(self.current_item.width), int(self.current_item.height), int(self.current_item.depth)),
            (int(self.current_item.depth), int(self.current_item.width), int(self.current_item.height)),
            (int(self.current_item.depth), int(self.current_item.height), int(self.current_item.width)),
            (int(self.current_item.height), int(self.current_item.width), int(self.current_item.depth)),
            (int(self.current_item.height), int(self.current_item.depth), int(self.current_item.width)),
        ]
        
        W, D, H = self.container.w, self.container.d, self.container.h
        max_actions = self._get_max_actions()
        action_feat_dim = self._get_action_feat_dim()
        
        action_feats = np.zeros((max_actions, action_feat_dim), dtype=np.float32)
        mask = np.zeros(max_actions, dtype=np.float32)
        
        action_idx = 0
        for ep in eps:
            for orientation in orientations:
                if action_idx >= max_actions:
                    break
                
                # EP features (normalized)
                ep_x, ep_y, ep_z = ep
                feat_ep_x = ep_x / W
                feat_ep_y = ep_y / D
                feat_ep_z = ep_z / H
                
                # Orientation features (normalized)
                w, d, h = orientation
                feat_w = w / W
                feat_d = d / D
                feat_h = h / H
                feat_vol = (w * d * h) / (W * D * H)
                
                # Feasibility
                is_feasible = self.container.can_place_at(
                    ep, orientation,
                    self.current_item.weight,
                    self.current_item.category
                )
                
                # Build feature vector
                action_feats[action_idx] = np.array([
                    feat_ep_x, feat_ep_y, feat_ep_z,
                    feat_w, feat_d, feat_h,
                    feat_vol,
                    float(is_feasible),
                ], dtype=np.float32)
                
                mask[action_idx] = 1.0 if is_feasible else 0.0
                action_idx += 1
        
        return action_feats, mask
    
    def _calculate_reward(self, success: bool) -> float:
        """Calculate reward for placement"""
        if not success:
            # Penalty for failed placement
            return -1.0
        
        # Reward for successful placement
        item = self.placed_items[-1]  # Just placed
        item_volume = item.width * item.depth * item.height
        container_volume = self.container.w * self.container.d * self.container.h
        
        # Base reward: volume efficiency
        reward = (item_volume / container_volume) * 10.0
        
        # Bonus for good weight utilization
        weight_util = self.container.current_weight / self.container.max_weight
        if 0.6 < weight_util < 0.95:
            reward += 0.5
        
        # Bonus for low EP count (compact packing)
        if len(self.container.eps) < 10:
            reward += 0.2
        
        return reward
    
    def _get_volume_utilization(self) -> float:
        """Calculate volume utilization"""
        if not self.container.placed:
            return 0.0
        used_volume = sum(b.w * b.d * b.h for b in self.container.placed)
        total_volume = self.container.w * self.container.d * self.container.h
        return used_volume / total_volume
    
    def _get_obs_dim(self) -> int:
        """Get observation dimension"""
        return 19  # See _encode_state
    
    def _get_max_actions(self) -> int:
        """Maximum number of actions (EP x orientations)"""
        return 100 * 6  # Max 100 EPs x 6 orientations
    
    def _get_action_feat_dim(self) -> int:
        """Action feature dimension"""
        return 8  # See _encode_actions


def train_dqn_q4realbpp(instance: Q4RealBPPInstance,
                        episodes: int = 500,
                        device: str = "cpu",
                        save_path: Optional[str] = None) -> DQNAgent:
    """
    Train DQN agent on Q4RealBPP instance.
    """
    # Create environment
    env = Q4RealBPPEnvironment(instance)
    
    # DQN config
    cfg = DQNConfig(
        obs_dim=env._get_obs_dim(),
        action_feat_dim=env._get_action_feat_dim(),
        max_actions=env._get_max_actions(),
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * 50,  # Decay over training
        buffer_size=100_000,
        n_step=3,
        target_update_interval=500,
        hidden=256,
        enc_layers=2,
        head_hidden=256,
        device=device,
        double_dqn=True,
        warmup_steps=1000,
    )
    
    agent = DQNAgent(cfg)
    
    # Training loop
    best_items_placed = 0
    best_volume_util = 0.0
    
    for episode in range(episodes):
        obs, action_feats, mask = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done:
            # Select action
            action_idx = agent.select_action(obs, action_feats, mask)
            if action_idx is None:
                # No valid actions
                break
            
            # Take step
            (next_obs, next_feats, next_mask), reward, done, info = env.step(action_idx)
            episode_reward += reward
            
            # Store transition
            agent.store(
                obs, action_idx, reward, next_obs, done,
                curr_action_feats=action_feats,
                curr_mask=mask,
                next_action_feats=next_feats,
                next_mask=next_mask
            )
            
            # Train
            loss = agent.train_step()
            
            # Update
            obs = next_obs
            action_feats = next_feats
            mask = next_mask
            step += 1
        
        # Episode stats
        items_placed = info['items_placed']
        volume_util = info['volume_utilization']
        
        if items_placed > best_items_placed:
            best_items_placed = items_placed
        if volume_util > best_volume_util:
            best_volume_util = volume_util
        
        if (episode + 1) % 10 == 0:
            eps = agent.epsilon()
            print(f"Episode {episode+1}/{episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Items={items_placed}/{len(instance.items)}, "
                  f"VolumeUtil={volume_util:.2%}, "
                  f"Eps={eps:.3f}")
    
    print(f"\nTraining complete!")
    print(f"Best items placed: {best_items_placed}/{len(instance.items)}")
    print(f"Best volume utilization: {best_volume_util:.2%}")
    
    if save_path:
        agent.save(save_path)
        print(f"Agent saved to {save_path}")
    
    return agent


def evaluate_dqn_q4realbpp(agent: DQNAgent, 
                           instance: Q4RealBPPInstance,
                           num_eval: int = 10,
                           visualize: bool = False) -> Dict:
    """Evaluate trained DQN agent"""
    env = Q4RealBPPEnvironment(instance)
    
    results = []
    
    for i in range(num_eval):
        obs, action_feats, mask = env.reset()
        done = False
        episode_reward = 0.0
        
        # Greedy evaluation (no exploration)
        agent._eps = 0.0
        
        while not done:
            action_idx = agent.select_action(obs, action_feats, mask)
            if action_idx is None:
                break
            
            (next_obs, next_feats, next_mask), reward, done, info = env.step(action_idx)
            episode_reward += reward
            obs, action_feats, mask = next_obs, next_feats, next_mask
        
        results.append({
            'items_placed': info['items_placed'],
            'volume_utilization': info['volume_utilization'],
            'weight_used': info['weight_used'],
            'reward': episode_reward,
        })
        
        if visualize and i == 0:
            # Visualize first solution
            env.container.plot3d(title=f"DQN Solution - {info['items_placed']} items")
    
    # Aggregate results
    avg_items = np.mean([r['items_placed'] for r in results])
    avg_volume = np.mean([r['volume_utilization'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    
    print(f"\nEvaluation Results ({num_eval} episodes):")
    print(f"  Avg items placed: {avg_items:.1f}/{len(instance.items)}")
    print(f"  Avg volume utilization: {avg_volume:.2%}")
    print(f"  Avg reward: {avg_reward:.2f}")
    
    return {
        'avg_items_placed': avg_items,
        'avg_volume_utilization': avg_volume,
        'avg_reward': avg_reward,
        'all_results': results,
    }


def main():
    """Example: Train DQN on Q4RealBPP instance"""
    print("="*60)
    print("DQN for Q4RealBPP 3D Bin Packing")
    print("="*60)
    
    # Load or generate instance
    print("\nGenerating Q4RealBPP instance...")
    instance = Q4RealBPPLoader.generate_sample_instance(num_items=20, num_bins=1)
    print(f"Instance: {instance}")
    
    # Train
    print(f"\nTraining DQN agent...")
    agent = train_dqn_q4realbpp(
        instance,
        episodes=200,
        device="cpu",
        save_path="dqn_q4realbpp.pth"
    )
    
    # Evaluate
    print(f"\nEvaluating agent...")
    results = evaluate_dqn_q4realbpp(agent, instance, num_eval=5, visualize=False)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
