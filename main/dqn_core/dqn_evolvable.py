"""
Extended DQN components for neurogenesis support.

This file extends the base DQN implementation to support evolved architectures
from the neurogenesis system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
from dqn_base import DQNConfig as BaseDQNConfig, QNetwork as BaseQNetwork


@dataclass
class EvolvableDQNConfig(BaseDQNConfig):
    """
    Extended DQN configuration that supports evolved architectures.
    
    This adds parameters for flexible network architectures that can be
    evolved by the neurogenesis system.
    """
    # Evolved architecture parameters
    state_encoder_layers: List[int] = field(default_factory=lambda: [256])
    action_encoder_layers: List[int] = field(default_factory=lambda: [256])
    head_layers: List[int] = field(default_factory=lambda: [128])
    activation: str = 'relu'
    
    def __post_init__(self):
        """Ensure backwards compatibility with base config"""
        # If using old-style hidden parameter, convert to new style
        if not hasattr(self, 'state_encoder_layers') or not self.state_encoder_layers:
            self.state_encoder_layers = [self.hidden]
        if not hasattr(self, 'action_encoder_layers') or not self.action_encoder_layers:
            self.action_encoder_layers = [self.hidden]
        if not hasattr(self, 'head_layers') or not self.head_layers:
            self.head_layers = [self.head_hidden]


class EvolvableMLP(nn.Module):
    """
    Flexible MLP that can be configured with arbitrary layer sizes.
    Supports the evolved architectures from neurogenesis.
    """
    def __init__(self, 
                 in_dim: int,
                 layer_sizes: List[int],
                 activation: str = 'relu'):
        super().__init__()
        
        # Get activation function
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }
        act_fn = activations.get(activation.lower(), nn.ReLU)
        
        # Build layers
        layers = []
        prev_size = in_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(act_fn())
            prev_size = size
        
        self.net = nn.Sequential(*layers)
        self.out_dim = layer_sizes[-1] if layer_sizes else in_dim
    
    def forward(self, x):
        return self.net(x)


class EvolvableHead(nn.Module):
    """
    Flexible Q-value head that supports evolved architectures.
    """
    def __init__(self,
                 in_dim: int,
                 layer_sizes: List[int],
                 activation: str = 'relu'):
        super().__init__()
        
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }
        act_fn = activations.get(activation.lower(), nn.ReLU)
        
        # Build head layers
        layers = []
        prev_size = in_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(act_fn())
            prev_size = size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class EvolvableQNetwork(nn.Module):
    """
    Q-Network that supports evolved architectures from neurogenesis.
    
    This can be instantiated with:
    1. Old-style parameters (for backwards compatibility)
    2. EvolvableDQNConfig (for evolved architectures)
    """
    def __init__(self,
                 obs_dim: Optional[int] = None,
                 action_feat_dim: Optional[int] = None,
                 hidden: int = 256,
                 enc_layers: int = 2,
                 head_hidden: int = 256,
                 config: Optional[EvolvableDQNConfig] = None):
        super().__init__()
        
        if config is not None:
            # New style: use config
            obs_dim = config.obs_dim
            action_feat_dim = config.action_feat_dim
            state_layers = config.state_encoder_layers
            action_layers = config.action_encoder_layers
            head_layers = config.head_layers
            activation = config.activation
        else:
            # Old style: backward compatible
            state_layers = [hidden] * (enc_layers - 1) + [hidden]
            action_layers = [hidden] * (enc_layers - 1) + [hidden]
            head_layers = [head_hidden]
            activation = 'relu'
        
        # State encoder
        self.state_enc = EvolvableMLP(
            in_dim=obs_dim,
            layer_sizes=state_layers,
            activation=activation
        )
        
        # Action encoder  
        self.action_enc = EvolvableMLP(
            in_dim=action_feat_dim,
            layer_sizes=action_layers,
            activation=activation
        )
        
        # Q-value head
        combined_dim = self.state_enc.out_dim + self.action_enc.out_dim
        self.head = EvolvableHead(
            in_dim=combined_dim,
            layer_sizes=head_layers,
            activation=activation
        )
    
    def forward(self, s: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            s: State observations [B, obs_dim]
            action_feats: Action features [B, max_actions, action_feat_dim]
            
        Returns:
            Q-values [B, max_actions]
        """
        B, A, D = action_feats.shape
        
        # Encode state
        zs = self.state_enc(s)  # [B, hidden]
        
        # Encode actions
        za = self.action_enc(action_feats.view(B * A, D))  # [B*A, hidden]
        
        # Combine state and action representations
        zs_rep = zs.unsqueeze(1).repeat(1, A, 1).view(B * A, -1)
        z = torch.cat([zs_rep, za], dim=-1)  # [B*A, 2*hidden]
        
        # Compute Q-values
        q = self.head(z).view(B, A)  # [B, A]
        
        return q
    
    def get_architecture_info(self) -> dict:
        """Get information about the network architecture"""
        return {
            'state_encoder_layers': [self.state_enc.out_dim],
            'action_encoder_layers': [self.action_enc.out_dim],
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# WEIGHT TRANSFER UTILITIES
# ============================================================================

def transfer_weights_smart(old_network: nn.Module,
                           new_network: nn.Module,
                           verbose: bool = False) -> None:
    """
    Transfer weights from old network to new network intelligently.
    
    This is crucial when evolving architectures - we want to preserve
    learned knowledge when possible.
    
    Strategy:
    - For layers with matching dimensions, copy exactly
    - For layers with different dimensions, copy what fits and initialize the rest
    
    Paper mentions: "the direct encoding scheme of network architectures is very 
    good at generating a compact architecture, while the indirect encoding scheme 
    is suitable for finding a particular type of network architecture quickly."
    
    This weight transfer helps bridge architecture changes during evolution.
    """
    old_params = dict(old_network.named_parameters())
    new_params = dict(new_network.named_parameters())
    
    transferred = 0
    initialized = 0
    
    for name, new_param in new_params.items():
        if name in old_params:
            old_param = old_params[name]
            
            # Check if shapes match
            if old_param.shape == new_param.shape:
                # Exact match - copy directly
                new_param.data.copy_(old_param.data)
                transferred += 1
            else:
                # Partial match - copy what fits
                if old_param.dim() == new_param.dim():
                    # Get minimum dimensions
                    if old_param.dim() == 2:  # Weight matrix
                        min_out = min(old_param.shape[0], new_param.shape[0])
                        min_in = min(old_param.shape[1], new_param.shape[1])
                        new_param.data[:min_out, :min_in].copy_(
                            old_param.data[:min_out, :min_in]
                        )
                    elif old_param.dim() == 1:  # Bias vector
                        min_size = min(old_param.shape[0], new_param.shape[0])
                        new_param.data[:min_size].copy_(old_param.data[:min_size])
                    transferred += 1
                else:
                    initialized += 1
        else:
            initialized += 1
    
    if verbose:
        print(f"Weight transfer: {transferred} layers transferred, "
              f"{initialized} layers initialized")


# ============================================================================
# EXTENDED DQN AGENT
# ============================================================================

from dqn_base import DQNAgent, ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class EvolvableDQNAgent(DQNAgent):
    """
    Extended DQN Agent that supports architecture evolution.
    
    Key differences from base agent:
    - Can update its architecture during training
    - Transfers weights when architecture changes
    - Tracks architecture history
    """
    
    def __init__(self, cfg: EvolvableDQNConfig):
        # Initialize with evolvable network
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Build evolvable networks
        self.q = EvolvableQNetwork(config=cfg).to(self.device)
        self.q_target = EvolvableQNetwork(config=cfg).to(self.device)
        
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        
        self.buffer = ReplayBuffer(
            cfg.buffer_size, cfg.obs_dim, cfg.max_actions,
            cfg.action_feat_dim, n_step=cfg.n_step, gamma=cfg.gamma
        )
        
        self.env_steps = 0
        self.training_steps = 0
        self._eps = cfg.eps_start
        
        # Track architecture changes
        self.architecture_history = []
    
    def update_architecture(self, 
                          new_config: EvolvableDQNConfig,
                          transfer_weights: bool = True) -> None:
        """
        Update the agent's architecture.
        
        This is the key method for neurogenesis - it allows changing
        the network architecture during evolution while preserving
        learned knowledge.
        
        Args:
            new_config: New configuration with evolved architecture
            transfer_weights: Whether to transfer weights from old network
        """
        # Save old network
        old_q = self.q
        old_q_target = self.q_target
        
        # Build new networks
        new_q = EvolvableQNetwork(config=new_config).to(self.device)
        new_q_target = EvolvableQNetwork(config=new_config).to(self.device)
        
        # Transfer weights if requested
        if transfer_weights:
            transfer_weights_smart(old_q, new_q, verbose=False)
            transfer_weights_smart(old_q_target, new_q_target, verbose=False)
        
        # Update networks
        self.q = new_q
        self.q_target = new_q_target
        self.q_target.eval()
        
        # Reinitialize optimizer with new parameters
        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        
        # Update config
        self.cfg = new_config
        
        # Track change
        self.architecture_history.append({
            'step': self.training_steps,
            'architecture': self.q.get_architecture_info()
        })
    
    def get_architecture_history(self) -> list:
        """Get history of architecture changes"""
        return self.architecture_history.copy()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create agent with evolved architecture
    
    config = EvolvableDQNConfig(
        obs_dim=10,
        action_feat_dim=5,
        max_actions=4,
        state_encoder_layers=[128, 64],
        action_encoder_layers=[128, 64],
        head_layers=[64, 32],
        activation='relu',
        device='cpu'
    )
    
    agent = EvolvableDQNAgent(config)
    print("Agent created with evolved architecture")
    print(f"Architecture info: {agent.q.get_architecture_info()}")
    
    # Simulate architecture update
    new_config = EvolvableDQNConfig(
        obs_dim=10,
        action_feat_dim=5,
        max_actions=4,
        state_encoder_layers=[256, 128, 64],  # Deeper network
        action_encoder_layers=[256, 128, 64],
        head_layers=[128, 64],
        activation='elu',
        device='cpu'
    )
    
    agent.update_architecture(new_config, transfer_weights=True)
    print("\nArchitecture updated")
    print(f"New architecture info: {agent.q.get_architecture_info()}")
