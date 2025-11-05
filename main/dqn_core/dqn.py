from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy
import random

def to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.as_tensor(x, device=device)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, max_actions: int, action_feat_dim: int, n_step:int=1, gamma:float=0.99):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.action_feat_dim = action_feat_dim
        self.n_step = max(1, n_step)
        self.gamma = gamma

        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a_idx = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        A, D = max_actions, action_feat_dim
        self.curr_feats = np.zeros((capacity, A, D), dtype=np.float32)
        self.curr_mask  = np.zeros((capacity, A), dtype=np.float32)
        self.next_feats = np.zeros((capacity, A, D), dtype=np.float32)
        self.next_mask  = np.zeros((capacity, A), dtype=np.float32)

        self.n_step_buffer = deque()

    def _n_step_push(self, transition):
        """
        Maintain sliding window and compute n-step return
        """
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return None
        
        oldest = self.n_step_buffer[0]
        s = oldest[0]
        a_idx = oldest[1]
        currF = oldest[5]
        currM = oldest[6]
        
        R = 0.0
        gamma_power = 1.0
        
        for i, (si, ai, ri, sni, di, cF, cM, nF, nM) in enumerate(self.n_step_buffer):
            R += gamma_power * ri
            gamma_power *= self.gamma
            
            if di:
                self.n_step_buffer.popleft()
                return (s, a_idx, R, sni, 1.0, currF, currM, nF, nM)
        
        last = self.n_step_buffer[-1]
        s_next = last[3]
        done = last[4]
        nextF = last[7]
        nextM = last[8]
        
        self.n_step_buffer.popleft()
        
        return (s, a_idx, R, s_next, float(done), currF, currM, nextF, nextM)

    def push(self, s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask):
        if self.n_step > 1:
            agg = self._n_step_push((s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask))
            if agg is None:
                return
            s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask = agg

        i = self.ptr
        self.s[i] = s
        self.a_idx[i] = -1 if a_idx is None else int(a_idx)
        self.r[i] = r
        self.s_next[i] = s_next
        self.done[i] = float(done)
        self.curr_feats[i] = curr_action_feats
        self.curr_mask[i]  = curr_mask
        self.next_feats[i] = next_action_feats
        self.next_mask[i]  = next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size:int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s = self.s[idxs],
            a_idx = self.a_idx[idxs],
            r = self.r[idxs],
            s_next = self.s_next[idxs],
            done = self.done[idxs],
            curr_feats = self.curr_feats[idxs],
            curr_mask = self.curr_mask[idxs],
            next_feats = self.next_feats[idxs],
            next_mask = self.next_mask[idxs],
        )


# ============================================================================
# EVOLVED ARCHITECTURE COMPONENTS
# ============================================================================

def get_activation(name: str) -> nn.Module:
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
    }
    return activations.get(name.lower(), nn.ReLU)


class MLP(nn.Module):
    """
    Multi-layer perceptron with backward-compatible interface.
    Can accept either old-style (hidden, layers) or new-style (layer_sizes list).
    """
    def __init__(self, 
                 in_dim: int = None,
                 hidden: int = 256, 
                 out_dim: int = 256, 
                 layers: int = 2, 
                 activation = nn.ReLU,
                 layer_sizes: Optional[List[int]] = None,
                 activation_name: str = 'relu'):
        super().__init__()
        
        # NEW STYLE: layer_sizes provided
        if layer_sizes is not None:
            sizes = [in_dim] + layer_sizes if in_dim is not None else layer_sizes
            act_fn = get_activation(activation_name) if isinstance(activation_name, str) else activation
            
            mods = []
            for i in range(len(sizes) - 1):
                mods.append(nn.Linear(sizes[i], sizes[i+1]))
                if i < len(sizes) - 2:  # No activation after last layer
                    mods.append(act_fn())
            self.net = nn.Sequential(*mods)
        
        # OLD STYLE: backward compatible
        else:
            mods = []
            d = in_dim
            for _ in range(max(0, layers - 1)):
                mods += [nn.Linear(d, hidden), activation()]
                d = hidden
            mods += [nn.Linear(d, out_dim)]
            self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class SimpleHead(nn.Module):
    """Q-value head - can use old or new style"""
    def __init__(self, 
                 in_dim: int,
                 hidden: int = 256,
                 layer_sizes: Optional[List[int]] = None,
                 activation_name: str = 'relu'):
        super().__init__()
        
        if layer_sizes is not None:
            # New style: custom layers
            act_fn = get_activation(activation_name)
            sizes = [in_dim] + layer_sizes + [1]
            mods = []
            for i in range(len(sizes) - 1):
                mods.append(nn.Linear(sizes[i], sizes[i+1]))
                if i < len(sizes) - 2:  # No activation after final output
                    mods.append(act_fn())
            self.net = nn.Sequential(*mods)
        else:
            # Old style: backward compatible
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
    
    def forward(self, z):
        return self.net(z)


class QNetwork(nn.Module):
    """
    Q-Network with dual encoder architecture.
    Backward compatible with old constructor, but also accepts DQNConfig.
    """
    def __init__(self, 
                 obs_dim: int = None,
                 action_feat_dim: int = None,
                 hidden: int = 256,
                 enc_layers: int = 2,
                 head_hidden: int = 256,
                 config: Optional['DQNConfig'] = None):
        super().__init__()
        
        if config is not None:
            # New style: build from config genotype
            arch = config.get_architecture()
            
            self.state_enc = MLP(
                in_dim=config.obs_dim,
                layer_sizes=arch['state_enc_layers'],
                activation_name=arch['state_enc_activation']
            )
            
            self.action_enc = MLP(
                in_dim=config.action_feat_dim,
                layer_sizes=arch['action_enc_layers'],
                activation_name=arch['action_enc_activation']
            )
            
            # Head input is concatenation of encoder outputs
            head_in = arch['state_enc_layers'][-1] + arch['action_enc_layers'][-1]
            self.head = SimpleHead(
                in_dim=head_in,
                layer_sizes=arch['head_layers'],
                activation_name=arch['head_activation']
            )
        else:
            # Old style: backward compatible
            self.state_enc = MLP(obs_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
            self.action_enc = MLP(action_feat_dim, hidden=hidden, out_dim=hidden, layers=enc_layers)
            self.head = SimpleHead(2*hidden, hidden=head_hidden)

    def forward(self, s: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        B, A, D = action_feats.shape
        zs = self.state_enc(s)                               # (B, H_s)
        za = self.action_enc(action_feats.view(B*A, D))      # (B*A, H_a)
        zs_rep = zs.unsqueeze(1).repeat(1, A, 1).view(B*A, -1)
        z = torch.cat([zs_rep, za], dim=-1)                  # (B*A, H_s+H_a)
        q = self.head(z).view(B, A)                          # (B, A)
        return q


# ============================================================================
# EVOLVABLE DQN CONFIG (TIER 1: PARAMETERIZATION CODING)
# ============================================================================

# ============================================================================
# TIER 2: DEVELOPMENTAL RULES (Pure Neurogenesis)
# ============================================================================

@dataclass
class DevelopmentalRule:
    """
    A generative rule that creates network architectures.
    This is the true GENOTYPE in Tier 2 - not the architecture itself.
    
    From the paper:
    "Developmental rule representation organizes the developmental rule into 
    the chromosome, evolves the rules continuously and generates suitable neural network."
    """
    rule_type: str          # 'linear', 'pyramid', 'hourglass', 'bottleneck', 'exponential'
    base_size: int          # Starting layer size
    depth: int              # Number of layers to generate
    growth_factor: float    # Multiplicative change per layer (e.g., 0.75 = shrink 25% per layer)
    size_step: int = 0      # Additive change per layer (alternative/addition to growth_factor)
    
    def __post_init__(self):
        """Validate rule parameters"""
        assert self.rule_type in ['linear', 'pyramid', 'hourglass', 'bottleneck', 'exponential'], \
            f"Invalid rule_type: {self.rule_type}"
        assert self.base_size > 0, "base_size must be positive"
        assert self.depth >= 1, "depth must be at least 1"
    
    def apply(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None) -> List[int]:
        """
        Apply the developmental rule to generate a list of layer sizes.
        
        Args:
            input_dim: Input dimension (for first layer constraint)
            output_dim: Output dimension (for last layer constraint)
            
        Returns:
            List of layer sizes
        """
        layers = []
        current_size = self.base_size
        
        if self.rule_type == 'linear':
            # Constant size: [256, 256, 256]
            layers = [self.base_size] * self.depth
        
        elif self.rule_type == 'pyramid':
            # Decreasing: [512, 384, 256, 192, 128]
            for i in range(self.depth):
                layers.append(int(current_size))
                current_size = current_size * self.growth_factor - self.size_step
                current_size = max(32, current_size)  # Floor
        
        elif self.rule_type == 'hourglass':
            # Compress then expand: [256, 128, 64, 128, 256]
            half = self.depth // 2
            # Compress
            for i in range(half):
                layers.append(int(current_size))
                current_size = current_size * self.growth_factor - self.size_step
                current_size = max(32, current_size)
            # Expand
            for i in range(self.depth - half):
                layers.append(int(current_size))
                current_size = current_size / self.growth_factor + self.size_step
                current_size = min(1024, current_size)  # Ceiling
        
        elif self.rule_type == 'bottleneck':
            # Expand then compress: [128, 256, 512, 256, 128]
            half = self.depth // 2
            # Expand
            for i in range(half):
                layers.append(int(current_size))
                current_size = current_size / self.growth_factor + self.size_step
                current_size = min(1024, current_size)
            # Compress
            for i in range(self.depth - half):
                layers.append(int(current_size))
                current_size = current_size * self.growth_factor - self.size_step
                current_size = max(32, current_size)
        
        elif self.rule_type == 'exponential':
            # Exponential decay/growth
            for i in range(self.depth):
                size = int(self.base_size * (self.growth_factor ** i))
                size = max(32, min(1024, size))
                layers.append(size)
        
        return layers
    
    def mutate(self, mutation_rate: float = 0.15, mutation_strength: float = 0.2) -> 'DevelopmentalRule':
        """
        Mutate the developmental rule.
        This is GENOTYPE-level mutation - changes how architectures are generated.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Magnitude of mutations
            
        Returns:
            New mutated rule
        """
        new_rule = copy.deepcopy(self)
        
        # Mutate rule type
        if random.random() < mutation_rate:
            new_rule.rule_type = random.choice(['linear', 'pyramid', 'hourglass', 'bottleneck', 'exponential'])
        
        # Mutate base size
        if random.random() < mutation_rate:
            change = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            new_rule.base_size = int(new_rule.base_size * change)
            new_rule.base_size = max(32, min(1024, new_rule.base_size))
        
        # Mutate depth
        if random.random() < mutation_rate:
            delta = random.choice([-1, 0, 1])
            new_rule.depth = max(1, new_rule.depth + delta)
        
        # Mutate growth factor
        if random.random() < mutation_rate:
            change = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            new_rule.growth_factor *= change
            new_rule.growth_factor = max(0.5, min(1.5, new_rule.growth_factor))
        
        # Mutate size step
        if random.random() < mutation_rate:
            delta = random.randint(-16, 16)
            new_rule.size_step = max(-64, min(64, new_rule.size_step + delta))
        
        return new_rule
    
    @staticmethod
    def crossover(parent1: 'DevelopmentalRule', parent2: 'DevelopmentalRule') -> 'DevelopmentalRule':
        """
        Crossover two developmental rules.
        
        Args:
            parent1, parent2: Parent rules
            
        Returns:
            Child rule (blended)
        """
        return DevelopmentalRule(
            rule_type=random.choice([parent1.rule_type, parent2.rule_type]),
            base_size=(parent1.base_size + parent2.base_size) // 2,
            depth=random.choice([parent1.depth, parent2.depth]),
            growth_factor=(parent1.growth_factor + parent2.growth_factor) / 2,
            size_step=(parent1.size_step + parent2.size_step) // 2
        )
    
    def __repr__(self):
        return (f"DevelopmentalRule(type={self.rule_type}, base={self.base_size}, "
                f"depth={self.depth}, growth={self.growth_factor:.2f})")


@dataclass
class DQNGenotype:
    """
    The true GENOTYPE for Tier 2 neurogenesis.
    Stores developmental RULES, not explicit architectures.
    
    From the paper:
    "Developmental rule representation can be use to design large-scale networks, 
    it has good regularity and generalization ability... capable of preserving 
    promising building blocks found so far."
    
    Usage:
        genotype = DQNGenotype(...)
        phenotype = genotype.express()  # Generate DQNConfig
        network = QNetwork(config=phenotype)
        # Train network from scratch (no weight transfer)
    """
    # Environment parameters (immutable)
    obs_dim: int
    action_feat_dim: int
    max_actions: int
    
    # Developmental rules (the actual genotype)
    state_enc_rule: DevelopmentalRule
    action_enc_rule: DevelopmentalRule
    head_rule: DevelopmentalRule
    
    # Activation genes
    state_enc_activation: str = 'relu'
    action_enc_activation: str = 'relu'
    head_activation: str = 'relu'
    
    # Hyperparameter genes
    lr: float = 2e-4
    gamma: float = 0.985
    
    # Evolution metadata
    generation: int = 0
    parent_ids: Tuple[int, ...] = field(default_factory=tuple)
    
    # Training parameters (typically not evolved)
    batch_size: int = 64
    grad_clip: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    buffer_size: int = 200_000
    n_step: int = 1
    target_update_interval: int = 1_000
    tau: float = 0.0
    device: str = "cpu"
    double_dqn: bool = True
    warmup_steps: int = 1000
    
    @classmethod
    def create_default(cls, obs_dim: int, action_feat_dim: int, max_actions: int) -> 'DQNGenotype':
        """Create a default genotype with reasonable rules"""
        return cls(
            obs_dim=obs_dim,
            action_feat_dim=action_feat_dim,
            max_actions=max_actions,
            state_enc_rule=DevelopmentalRule('pyramid', base_size=256, depth=3, growth_factor=0.75),
            action_enc_rule=DevelopmentalRule('pyramid', base_size=256, depth=2, growth_factor=0.75),
            head_rule=DevelopmentalRule('pyramid', base_size=256, depth=2, growth_factor=0.75)
        )
    
    def express(self) -> DQNConfig:
        """
        Express the genotype into a phenotype (architecture).
        This is the genotype → phenotype mapping.
        
        Returns:
            DQNConfig with architecture generated from rules
        """
        # Apply developmental rules to generate layer architectures
        state_layers = self.state_enc_rule.apply()
        action_layers = self.action_enc_rule.apply()
        
        # Head input dimension depends on encoder outputs
        head_input_dim = state_layers[-1] + action_layers[-1]
        head_layers = self.head_rule.apply()
        
        # Create phenotype (DQNConfig)
        return DQNConfig(
            obs_dim=self.obs_dim,
            action_feat_dim=self.action_feat_dim,
            max_actions=self.max_actions,
            state_enc_layers=state_layers,
            action_enc_layers=action_layers,
            head_layers=head_layers,
            state_enc_activation=self.state_enc_activation,
            action_enc_activation=self.action_enc_activation,
            head_activation=self.head_activation,
            lr=self.lr,
            gamma=self.gamma,
            batch_size=self.batch_size,
            grad_clip=self.grad_clip,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_decay_steps=self.eps_decay_steps,
            buffer_size=self.buffer_size,
            n_step=self.n_step,
            target_update_interval=self.target_update_interval,
            tau=self.tau,
            device=self.device,
            double_dqn=self.double_dqn,
            warmup_steps=self.warmup_steps,
            generation=self.generation,
            parent_ids=self.parent_ids
        )
    
    def mutate(self, mutation_rate: float = 0.15, mutation_strength: float = 0.2) -> 'DQNGenotype':
        """
        Mutate the genotype (the RULES, not the architecture).
        This is pure neurogenesis - we mutate the genome, not the network.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Magnitude of mutations
            
        Returns:
            New mutated genotype
        """
        new_genotype = copy.deepcopy(self)
        new_genotype.generation = self.generation + 1
        new_genotype.parent_ids = (id(self),)
        
        # Mutate developmental rules
        new_genotype.state_enc_rule = self.state_enc_rule.mutate(mutation_rate, mutation_strength)
        new_genotype.action_enc_rule = self.action_enc_rule.mutate(mutation_rate, mutation_strength)
        new_genotype.head_rule = self.head_rule.mutate(mutation_rate, mutation_strength)
        
        # Mutate activations
        if random.random() < mutation_rate:
            new_genotype.state_enc_activation = random.choice(['relu', 'tanh', 'elu', 'gelu'])
        if random.random() < mutation_rate:
            new_genotype.action_enc_activation = random.choice(['relu', 'tanh', 'elu', 'gelu'])
        if random.random() < mutation_rate:
            new_genotype.head_activation = random.choice(['relu', 'tanh', 'elu'])
        
        # Mutate learning rate
        if random.random() < mutation_rate:
            new_genotype.lr *= random.uniform(1 - mutation_strength, 1 + mutation_strength)
            new_genotype.lr = max(1e-5, min(1e-2, new_genotype.lr))
        
        # Mutate gamma
        if random.random() < mutation_rate * 0.5:  # Less frequent
            new_genotype.gamma *= random.uniform(1 - mutation_strength * 0.5, 1 + mutation_strength * 0.5)
            new_genotype.gamma = max(0.9, min(0.999, new_genotype.gamma))
        
        return new_genotype
    
    @staticmethod
    def crossover(parent1: 'DQNGenotype', parent2: 'DQNGenotype') -> Tuple['DQNGenotype', 'DQNGenotype']:
        """
        Crossover two genotypes (the RULES, not architectures).
        
        Args:
            parent1, parent2: Parent genotypes
            
        Returns:
            Two child genotypes
        """
        # Must have same environment parameters
        assert parent1.obs_dim == parent2.obs_dim
        assert parent1.action_feat_dim == parent2.action_feat_dim
        assert parent1.max_actions == parent2.max_actions
        
        # Create children
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        child1.generation = max(parent1.generation, parent2.generation) + 1
        child2.generation = child1.generation
        child1.parent_ids = (id(parent1), id(parent2))
        child2.parent_ids = (id(parent1), id(parent2))
        
        # Crossover rules (swap entire rules or blend them)
        if random.random() < 0.5:
            # Swap state encoder rules
            child1.state_enc_rule, child2.state_enc_rule = child2.state_enc_rule, child1.state_enc_rule
        else:
            # Blend state encoder rules
            child1.state_enc_rule = DevelopmentalRule.crossover(parent1.state_enc_rule, parent2.state_enc_rule)
            child2.state_enc_rule = DevelopmentalRule.crossover(parent2.state_enc_rule, parent1.state_enc_rule)
        
        if random.random() < 0.5:
            child1.action_enc_rule, child2.action_enc_rule = child2.action_enc_rule, child1.action_enc_rule
        else:
            child1.action_enc_rule = DevelopmentalRule.crossover(parent1.action_enc_rule, parent2.action_enc_rule)
            child2.action_enc_rule = DevelopmentalRule.crossover(parent2.action_enc_rule, parent1.action_enc_rule)
        
        if random.random() < 0.5:
            child1.head_rule, child2.head_rule = child2.head_rule, child1.head_rule
        else:
            child1.head_rule = DevelopmentalRule.crossover(parent1.head_rule, parent2.head_rule)
            child2.head_rule = DevelopmentalRule.crossover(parent2.head_rule, parent1.head_rule)
        
        # Crossover activations
        if random.random() < 0.5:
            child1.state_enc_activation, child2.state_enc_activation = \
                child2.state_enc_activation, child1.state_enc_activation
        
        # Blend hyperparameters
        avg_lr = (parent1.lr + parent2.lr) / 2
        child1.lr = avg_lr * random.uniform(0.9, 1.1)
        child2.lr = avg_lr * random.uniform(0.9, 1.1)
        
        avg_gamma = (parent1.gamma + parent2.gamma) / 2
        child1.gamma = max(0.9, min(0.999, avg_gamma))
        child2.gamma = max(0.9, min(0.999, avg_gamma))
        
        return child1, child2
    
    def complexity_penalty(self, penalty_weight: float = 1e-6) -> float:
        """
        Compute complexity penalty based on expressed phenotype.
        
        Args:
            penalty_weight: Weight for parameter count penalty
            
        Returns:
            Penalty value
        """
        phenotype = self.express()
        return phenotype.complexity_penalty(penalty_weight)
    
    def __repr__(self):
        return (f"DQNGenotype(gen={self.generation}, "
                f"state={self.state_enc_rule}, "
                f"action={self.action_enc_rule}, "
                f"head={self.head_rule})")


# ============================================================================
# TIER 1: PARAMETERIZATION CODING (Backward Compatible)
# ============================================================================

@dataclass
class DQNConfig:
    """
    DQN configuration with evolution capabilities.
    
    BACKWARD COMPATIBLE: Old code using (hidden, enc_layers, head_hidden) still works.
    NEW EVOLUTION: Use layer lists for fine-grained architecture control.
    TIER 2 READY: Can be generated from DQNGenotype later.
    """
    # Environment-specific (immutable)
    obs_dim: int
    action_feat_dim: int
    max_actions: int
    
    # Training hyperparameters
    gamma: float = 0.985
    lr: float = 2e-4
    batch_size: int = 64
    grad_clip: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    buffer_size: int = 200_000
    n_step: int = 1
    target_update_interval: int = 1_000
    tau: float = 0.0
    device: str = "cpu"
    double_dqn: bool = True
    warmup_steps: int = 1000
    
    # OLD STYLE (backward compatible, used if *_layers fields are None)
    hidden: int = 256
    enc_layers: int = 2
    head_hidden: int = 256
    
    # NEW STYLE (evolvable architecture - Tier 1 parameterization)
    state_enc_layers: Optional[List[int]] = None
    action_enc_layers: Optional[List[int]] = None
    head_layers: Optional[List[int]] = None
    
    # Activation functions per component
    state_enc_activation: str = 'relu'
    action_enc_activation: str = 'relu'
    head_activation: str = 'relu'
    
    # Evolution metadata
    generation: int = 0
    parent_ids: Tuple[int, ...] = field(default_factory=tuple)
    
    # Evolution constraints (configurable)
    min_layers: int = 1
    max_layers: int = 10  # Increased from 5, but still configurable
    min_layer_size: int = 32
    max_layer_size: int = 1024
    allow_unbounded_growth: bool = False  # Set True for true neurogenesis
    
    def __post_init__(self):
        """Auto-convert old style to new style if needed"""
        if self.state_enc_layers is None:
            # Convert old params to new format
            self.state_enc_layers = [self.hidden] * self.enc_layers
        if self.action_enc_layers is None:
            self.action_enc_layers = [self.hidden] * self.enc_layers
        if self.head_layers is None:
            self.head_layers = [self.head_hidden]
    
    def get_architecture(self) -> dict:
        """Get the resolved architecture specification"""
        return {
            'state_enc_layers': self.state_enc_layers,
            'action_enc_layers': self.action_enc_layers,
            'head_layers': self.head_layers,
            'state_enc_activation': self.state_enc_activation,
            'action_enc_activation': self.action_enc_activation,
            'head_activation': self.head_activation,
        }
    
    def count_parameters(self) -> int:
        """Estimate total trainable parameters"""
        # State encoder
        params = 0
        prev = self.obs_dim
        for size in self.state_enc_layers:
            params += prev * size + size  # weights + bias
            prev = size
        
        # Action encoder
        prev = self.action_feat_dim
        for size in self.action_enc_layers:
            params += prev * size + size
            prev = size
        
        # Head
        prev = self.state_enc_layers[-1] + self.action_enc_layers[-1]
        for size in self.head_layers:
            params += prev * size + size
            prev = size
        params += prev * 1 + 1  # Final Q-value output
        
        return params
    
    def validate(self) -> bool:
        """Check if architecture is valid"""
        for layers in [self.state_enc_layers, self.action_enc_layers, self.head_layers]:
            # Check layer count
            if self.allow_unbounded_growth:
                if len(layers) < self.min_layers:
                    return False
                # No upper limit when unbounded growth is enabled
            else:
                if not (self.min_layers <= len(layers) <= self.max_layers):
                    return False
            
            # Check layer sizes
            for size in layers:
                if not (self.min_layer_size <= size <= self.max_layer_size):
                    return False
        
        return True
    
    # ========================================================================
    # EVOLUTION OPERATORS (TIER 1)
    # ========================================================================
    
    def mutate(self, 
               mutation_rate: float = 0.15,
               mutation_strength: float = 0.2) -> 'DQNConfig':
        """
        Mutate architecture and hyperparameters.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Magnitude of mutations (as fraction)
        
        Returns:
            New mutated config
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        # Mutate layer sizes
        new_cfg.state_enc_layers = self._mutate_layer_list(
            self.state_enc_layers, mutation_rate, mutation_strength
        )
        new_cfg.action_enc_layers = self._mutate_layer_list(
            self.action_enc_layers, mutation_rate, mutation_strength
        )
        new_cfg.head_layers = self._mutate_layer_list(
            self.head_layers, mutation_rate, mutation_strength
        )
        
        # Mutate activations
        if random.random() < mutation_rate:
            new_cfg.state_enc_activation = random.choice(['relu', 'tanh', 'elu', 'gelu'])
        if random.random() < mutation_rate:
            new_cfg.action_enc_activation = random.choice(['relu', 'tanh', 'elu', 'gelu'])
        if random.random() < mutation_rate:
            new_cfg.head_activation = random.choice(['relu', 'tanh', 'elu'])
        
        # Mutate learning rate
        if random.random() < mutation_rate:
            new_cfg.lr *= random.uniform(1 - mutation_strength, 1 + mutation_strength)
            new_cfg.lr = max(1e-5, min(1e-2, new_cfg.lr))
        
        # Validate and fix if needed
        if not new_cfg.validate():
            new_cfg = self._repair_config(new_cfg)
        
        return new_cfg
    
    def _mutate_layer_list(self, 
                          layers: List[int],
                          mutation_rate: float,
                          mutation_strength: float) -> List[int]:
        """Mutate a list of layer sizes"""
        new_layers = layers.copy()
        
        # Add or remove layer
        if random.random() < mutation_rate * 0.5:
            # Try to add layer
            can_add = self.allow_unbounded_growth or len(new_layers) < self.max_layers
            # Try to remove layer
            can_remove = len(new_layers) > self.min_layers
            
            if can_add and can_remove:
                # Both possible, choose randomly
                if random.random() < 0.5:
                    # Add layer
                    insert_pos = random.randint(0, len(new_layers))
                    if insert_pos == 0:
                        new_size = new_layers[0]
                    elif insert_pos == len(new_layers):
                        new_size = new_layers[-1]
                    else:
                        new_size = (new_layers[insert_pos-1] + new_layers[insert_pos]) // 2
                    new_layers.insert(insert_pos, new_size)
                else:
                    # Remove layer
                    remove_pos = random.randint(0, len(new_layers) - 1)
                    new_layers.pop(remove_pos)
            elif can_add:
                # Only adding possible
                insert_pos = random.randint(0, len(new_layers))
                if insert_pos == 0:
                    new_size = new_layers[0]
                elif insert_pos == len(new_layers):
                    new_size = new_layers[-1]
                else:
                    new_size = (new_layers[insert_pos-1] + new_layers[insert_pos]) // 2
                new_layers.insert(insert_pos, new_size)
            elif can_remove:
                # Only removal possible
                remove_pos = random.randint(0, len(new_layers) - 1)
                new_layers.pop(remove_pos)
        
        # Mutate layer sizes
        for i in range(len(new_layers)):
            if random.random() < mutation_rate:
                change = random.uniform(1 - mutation_strength, 1 + mutation_strength)
                new_layers[i] = int(new_layers[i] * change)
                new_layers[i] = max(self.min_layer_size, min(self.max_layer_size, new_layers[i]))
        
        return new_layers
    
    def _repair_config(self, cfg: 'DQNConfig') -> 'DQNConfig':
        """Fix invalid configuration"""
        # Clamp layer counts
        for attr in ['state_enc_layers', 'action_enc_layers', 'head_layers']:
            layers = getattr(cfg, attr)
            if len(layers) < cfg.min_layers:
                setattr(cfg, attr, [256] * cfg.min_layers)
            elif not cfg.allow_unbounded_growth and len(layers) > cfg.max_layers:
                setattr(cfg, attr, layers[:cfg.max_layers])
            
            # Clamp sizes
            clamped = [max(cfg.min_layer_size, min(cfg.max_layer_size, size)) for size in layers]
            setattr(cfg, attr, clamped)
        
        return cfg
    
    def complexity_penalty(self, penalty_weight: float = 1e-6) -> float:
        """
        Compute complexity penalty for fitness function.
        Use this to naturally limit growth instead of hard constraints.
        
        Args:
            penalty_weight: Weight for parameter count penalty
            
        Returns:
            Penalty value (higher = more complex)
            
        Example:
            fitness = reward - config.complexity_penalty(penalty_weight=1e-6)
        """
        return penalty_weight * self.count_parameters()
    
    # ========================================================================
    # ADVANCED NEUROGENESIS OPERATORS (Optional, for finer control)
    # ========================================================================
    
    def mutate_add_neurons(self, component: str = 'state_enc', 
                          layer_idx: Optional[int] = None,
                          num_neurons: int = 1) -> 'DQNConfig':
        """
        Add neurons to a specific layer (finer-grained than mutate).
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            layer_idx: Which layer to modify (None = random)
            num_neurons: How many neurons to add
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        # Get the appropriate layer list
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name)
        
        if layer_idx is None:
            layer_idx = random.randint(0, len(layers) - 1)
        
        layers[layer_idx] = min(self.max_layer_size, layers[layer_idx] + num_neurons)
        setattr(new_cfg, attr_name, layers)
        
        return new_cfg
    
    def mutate_remove_neurons(self, component: str = 'state_enc',
                             layer_idx: Optional[int] = None,
                             num_neurons: int = 1) -> 'DQNConfig':
        """
        Remove neurons from a specific layer (finer-grained than mutate).
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            layer_idx: Which layer to modify (None = random)
            num_neurons: How many neurons to remove
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name)
        
        if layer_idx is None:
            layer_idx = random.randint(0, len(layers) - 1)
        
        layers[layer_idx] = max(self.min_layer_size, layers[layer_idx] - num_neurons)
        setattr(new_cfg, attr_name, layers)
        
        return new_cfg
    
    def mutate_add_layer_strategic(self, component: str = 'state_enc',
                                   position: str = 'end',
                                   size: Optional[int] = None) -> 'DQNConfig':
        """
        Add a layer at a strategic position (more control than random).
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            position: 'start', 'end', 'middle', or 'random'
            size: Size of new layer (None = interpolate from neighbors)
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name).copy()
        
        # Check if we can add
        if not self.allow_unbounded_growth and len(layers) >= self.max_layers:
            return new_cfg  # Can't add, return unchanged
        
        # Determine position
        if position == 'start':
            insert_pos = 0
            default_size = layers[0] if layers else 256
        elif position == 'end':
            insert_pos = len(layers)
            default_size = layers[-1] if layers else 256
        elif position == 'middle':
            insert_pos = len(layers) // 2
            if insert_pos > 0 and insert_pos < len(layers):
                default_size = (layers[insert_pos-1] + layers[insert_pos]) // 2
            else:
                default_size = layers[insert_pos] if insert_pos < len(layers) else 256
        else:  # random
            insert_pos = random.randint(0, len(layers))
            if insert_pos == 0:
                default_size = layers[0] if layers else 256
            elif insert_pos == len(layers):
                default_size = layers[-1] if layers else 256
            else:
                default_size = (layers[insert_pos-1] + layers[insert_pos]) // 2
        
        new_size = size if size is not None else default_size
        layers.insert(insert_pos, new_size)
        setattr(new_cfg, attr_name, layers)
        
        return new_cfg
    
    def mutate_remove_layer_strategic(self, component: str = 'state_enc',
                                     criterion: str = 'smallest') -> 'DQNConfig':
        """
        Remove a layer based on a criterion (more control than random).
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            criterion: 'smallest', 'largest', 'first', 'last', or 'random'
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name).copy()
        
        if len(layers) <= self.min_layers:
            return new_cfg  # Can't remove, return unchanged
        
        # Determine which layer to remove
        if criterion == 'smallest':
            remove_idx = layers.index(min(layers))
        elif criterion == 'largest':
            remove_idx = layers.index(max(layers))
        elif criterion == 'first':
            remove_idx = 0
        elif criterion == 'last':
            remove_idx = len(layers) - 1
        else:  # random
            remove_idx = random.randint(0, len(layers) - 1)
        
        layers.pop(remove_idx)
        setattr(new_cfg, attr_name, layers)
        
        return new_cfg
    
    def mutate_split_layer(self, component: str = 'state_enc',
                          layer_idx: Optional[int] = None) -> 'DQNConfig':
        """
        Split a layer into two layers (increases depth while preserving capacity).
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            layer_idx: Which layer to split (None = random)
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name).copy()
        
        # Check if we can add
        if not self.allow_unbounded_growth and len(layers) >= self.max_layers:
            return new_cfg
        
        if layer_idx is None:
            layer_idx = random.randint(0, len(layers) - 1)
        
        # Split the layer into two smaller layers
        original_size = layers[layer_idx]
        split_size = original_size // 2
        
        layers[layer_idx] = split_size
        layers.insert(layer_idx + 1, split_size)
        
        setattr(new_cfg, attr_name, layers)
        return new_cfg
    
    def mutate_merge_layers(self, component: str = 'state_enc',
                           layer_idx: Optional[int] = None) -> 'DQNConfig':
        """
        Merge two adjacent layers into one (decreases depth, increases width).
        Inverse of split_layer.
        
        Args:
            component: 'state_enc', 'action_enc', or 'head'
            layer_idx: Index of first layer to merge (will merge with idx+1). None = random
        
        Returns:
            New config with merged layers
            
        Example:
            Before: [512, 256, 128, 64]
            After (merge idx=1): [512, 384, 64]  # 256+128=384
        """
        new_cfg = copy.deepcopy(self)
        new_cfg.generation = self.generation + 1
        new_cfg.parent_ids = (id(self),)
        
        attr_name = f"{component}_layers"
        layers = getattr(new_cfg, attr_name).copy()
        
        # Need at least 2 layers to merge
        if len(layers) < 2:
            return new_cfg
        
        # Check if we're at minimum layers
        if len(layers) <= self.min_layers:
            return new_cfg
        
        # Select which layers to merge
        if layer_idx is None:
            # Random layer (but not the last one, since we merge with idx+1)
            layer_idx = random.randint(0, len(layers) - 2)
        elif layer_idx >= len(layers) - 1:
            # Invalid index, can't merge last layer with nothing
            return new_cfg
        
        # Merge: combine sizes of adjacent layers
        merged_size = layers[layer_idx] + layers[layer_idx + 1]
        
        # Clamp to max_layer_size
        merged_size = min(merged_size, self.max_layer_size)
        
        # Replace first layer with merged size, remove second
        layers[layer_idx] = merged_size
        layers.pop(layer_idx + 1)
        
        setattr(new_cfg, attr_name, layers)
        return new_cfg
    
    @staticmethod
    def crossover(parent1: 'DQNConfig', 
                  parent2: 'DQNConfig',
                  method: str = 'uniform') -> Tuple['DQNConfig', 'DQNConfig']:
        """
        Crossover two parent configurations.
        
        Args:
            parent1, parent2: Parent configs
            method: 'uniform' (gene-wise), 'component' (module-wise), or 'blend'
        
        Returns:
            Two offspring configs
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        child1.generation = max(parent1.generation, parent2.generation) + 1
        child2.generation = child1.generation
        child1.parent_ids = (id(parent1), id(parent2))
        child2.parent_ids = (id(parent1), id(parent2))
        
        if method == 'uniform':
            # Randomly swap each component
            if random.random() < 0.5:
                child1.state_enc_layers, child2.state_enc_layers = \
                    child2.state_enc_layers, child1.state_enc_layers
            if random.random() < 0.5:
                child1.action_enc_layers, child2.action_enc_layers = \
                    child2.action_enc_layers, child1.action_enc_layers
            if random.random() < 0.5:
                child1.head_layers, child2.head_layers = \
                    child2.head_layers, child1.head_layers
            
            # Swap activations
            if random.random() < 0.5:
                child1.state_enc_activation, child2.state_enc_activation = \
                    child2.state_enc_activation, child1.state_enc_activation
            if random.random() < 0.5:
                child1.action_enc_activation, child2.action_enc_activation = \
                    child2.action_enc_activation, child1.action_enc_activation
            if random.random() < 0.5:
                child1.head_activation, child2.head_activation = \
                    child2.head_activation, child1.head_activation
            
            # Blend learning rate
            avg_lr = (parent1.lr + parent2.lr) / 2
            child1.lr = avg_lr * random.uniform(0.9, 1.1)
            child2.lr = avg_lr * random.uniform(0.9, 1.1)
        
        elif method == 'component':
            # Swap entire encoder modules
            if random.random() < 0.5:
                child1.state_enc_layers = parent2.state_enc_layers
                child1.state_enc_activation = parent2.state_enc_activation
                child2.state_enc_layers = parent1.state_enc_layers
                child2.state_enc_activation = parent1.state_enc_activation
            else:
                child1.action_enc_layers = parent2.action_enc_layers
                child1.action_enc_activation = parent2.action_enc_activation
                child2.action_enc_layers = parent1.action_enc_layers
                child2.action_enc_activation = parent1.action_enc_activation
        
        elif method == 'blend':
            # Average layer sizes where possible
            child1.state_enc_layers = DQNConfig._blend_layers(
                parent1.state_enc_layers, parent2.state_enc_layers
            )
            child2.state_enc_layers = DQNConfig._blend_layers(
                parent2.state_enc_layers, parent1.state_enc_layers
            )
            child1.action_enc_layers = DQNConfig._blend_layers(
                parent1.action_enc_layers, parent2.action_enc_layers
            )
            child2.action_enc_layers = DQNConfig._blend_layers(
                parent2.action_enc_layers, parent1.action_enc_layers
            )
            child1.head_layers = DQNConfig._blend_layers(
                parent1.head_layers, parent2.head_layers
            )
            child2.head_layers = DQNConfig._blend_layers(
                parent2.head_layers, parent1.head_layers
            )
        
        return child1, child2
    
    @staticmethod
    def _blend_layers(layers1: List[int], layers2: List[int]) -> List[int]:
        """Blend two layer lists by averaging sizes"""
        min_len = min(len(layers1), len(layers2))
        max_len = max(len(layers1), len(layers2))
        
        # Average common layers
        result = [(layers1[i] + layers2[i]) // 2 for i in range(min_len)]
        
        # Add remaining from longer list (with some randomness)
        if len(layers1) > min_len:
            result.extend(layers1[min_len:])
        elif len(layers2) > min_len:
            result.extend(layers2[min_len:])
        
        return result
    
    @staticmethod
    def transfer_weights(old_net: QNetwork,
                        new_net: QNetwork,
                        old_cfg: 'DQNConfig',
                        new_cfg: 'DQNConfig',
                        strategy: str = 'compatible') -> None:
        """
        Transfer weights from old network to new network after architecture change.
        
        Args:
            old_net: Source network
            new_net: Destination network
            old_cfg: Old configuration
            new_cfg: New configuration
            strategy: 'compatible' (only matching dims) or 'smart' (with padding/truncation)
        """
        with torch.no_grad():
            # Transfer state encoder
            DQNConfig._transfer_mlp_weights(
                old_net.state_enc,
                new_net.state_enc,
                old_cfg.state_enc_layers,
                new_cfg.state_enc_layers,
                strategy
            )
            
            # Transfer action encoder
            DQNConfig._transfer_mlp_weights(
                old_net.action_enc,
                new_net.action_enc,
                old_cfg.action_enc_layers,
                new_cfg.action_enc_layers,
                strategy
            )
            
            # Transfer head (more complex due to concatenation)
            old_head_in = old_cfg.state_enc_layers[-1] + old_cfg.action_enc_layers[-1]
            new_head_in = new_cfg.state_enc_layers[-1] + new_cfg.action_enc_layers[-1]
            
            DQNConfig._transfer_head_weights(
                old_net.head,
                new_net.head,
                old_head_in,
                new_head_in,
                old_cfg.head_layers,
                new_cfg.head_layers,
                strategy
            )
    
    @staticmethod
    def _transfer_mlp_weights(old_mlp: MLP,
                             new_mlp: MLP,
                             old_layers: List[int],
                             new_layers: List[int],
                             strategy: str) -> None:
        """Transfer weights between MLPs layer by layer"""
        old_params = list(old_mlp.net.parameters())
        new_params = list(new_mlp.net.parameters())
        
        # Filter out activation layers (no parameters)
        old_linear_params = [p for p in old_params if p.dim() >= 2]
        new_linear_params = [p for p in new_params if p.dim() >= 2]
        
        # Transfer layer by layer
        num_layers = min(len(old_layers), len(new_layers))
        for i in range(num_layers):
            if i * 2 >= len(old_linear_params) or i * 2 >= len(new_linear_params):
                break
            
            old_weight = old_linear_params[i * 2]      # Weight matrix
            old_bias = old_linear_params[i * 2 + 1]    # Bias vector
            new_weight = new_linear_params[i * 2]
            new_bias = new_linear_params[i * 2 + 1]
            
            if strategy == 'compatible':
                # Only copy if dimensions match exactly
                if old_weight.shape == new_weight.shape:
                    new_weight.copy_(old_weight)
                    new_bias.copy_(old_bias)
            
            elif strategy == 'smart':
                # Copy and pad/truncate as needed
                out_dim = min(old_weight.shape[0], new_weight.shape[0])
                in_dim = min(old_weight.shape[1], new_weight.shape[1])
                
                new_weight[:out_dim, :in_dim].copy_(old_weight[:out_dim, :in_dim])
                new_bias[:out_dim].copy_(old_bias[:out_dim])
    
    @staticmethod
    def _transfer_head_weights(old_head: SimpleHead,
                              new_head: SimpleHead,
                              old_in: int,
                              new_in: int,
                              old_layers: List[int],
                              new_layers: List[int],
                              strategy: str) -> None:
        """Transfer head weights (similar to MLP but handles input size change)"""
        old_params = list(old_head.net.parameters())
        new_params = list(new_head.net.parameters())
        
        old_linear = [p for p in old_params if p.dim() >= 2]
        new_linear = [p for p in new_params if p.dim() >= 2]
        
        num_layers = min(len(old_layers), len(new_layers))
        for i in range(num_layers):
            if i * 2 >= len(old_linear) or i * 2 >= len(new_linear):
                break
            
            old_weight = old_linear[i * 2]
            old_bias = old_linear[i * 2 + 1]
            new_weight = new_linear[i * 2]
            new_bias = new_linear[i * 2 + 1]
            
            if strategy == 'smart':
                out_dim = min(old_weight.shape[0], new_weight.shape[0])
                in_dim = min(old_weight.shape[1], new_weight.shape[1])
                
                new_weight[:out_dim, :in_dim].copy_(old_weight[:out_dim, :in_dim])
                new_bias[:out_dim].copy_(old_bias[:out_dim])


# ============================================================================
# DQN AGENT (Updated to use evolved configs)
# ============================================================================

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Initialize networks from evolved config
        self.q = QNetwork(config=cfg).to(self.device)
        self.q_target = QNetwork(config=cfg).to(self.device)
        
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        
        self.buffer = ReplayBuffer(
            cfg.buffer_size, cfg.obs_dim, cfg.max_actions, 
            cfg.action_feat_dim, n_step=cfg.n_step, gamma=cfg.gamma
        )
        
        self.env_steps = 0
        self.training_steps = 0
        self._eps = cfg.eps_start

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac
        return self._eps

    def select_action(self, obs: np.ndarray, action_feats: np.ndarray, mask: np.ndarray) -> int | None:
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return None
        
        eps = self.epsilon()
        self.env_steps += 1
        
        if np.random.rand() < eps:
            return int(np.random.choice(valid))
        
        obs_t = to_torch(obs, self.device).float().unsqueeze(0)
        feats_t = to_torch(action_feats, self.device).float().unsqueeze(0)
        
        with torch.no_grad():
            q = self.q(obs_t, feats_t)[0].cpu().numpy()
        
        q[mask < 0.5] = -np.inf
        return int(np.argmax(q))

    def store(self, s, a_idx, r, s_next, done, *, curr_action_feats, curr_mask, next_action_feats, next_mask):
        self.buffer.push(s, a_idx, r, s_next, done, curr_action_feats, curr_mask, next_action_feats, next_mask)

    def train_step(self) -> float | None:
        if self.buffer.size < self.cfg.warmup_steps:
            return None
        
        if self.buffer.size < self.cfg.batch_size:
            return None
        
        batch = self.buffer.sample(self.cfg.batch_size)
        s = to_torch(batch['s'], self.device).float()
        a_idx = to_torch(batch['a_idx'], self.device).long()
        r = to_torch(batch['r'], self.device).float()
        s_next = to_torch(batch['s_next'], self.device).float()
        done = to_torch(batch['done'], self.device).float()
        curr_feats = to_torch(batch['curr_feats'], self.device).float()
        curr_mask  = to_torch(batch['curr_mask'], self.device).float()
        next_feats = to_torch(batch['next_feats'], self.device).float()
        next_mask  = to_torch(batch['next_mask'], self.device).float()

        q_all = self.q(s, curr_feats)
        
        valid = (a_idx >= 0)
        if not valid.any():
            return None
        
        q_sa = q_all[valid].gather(1, a_idx[valid].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                q_next_online = self.q(s_next, next_feats)
                q_next_online_masked = torch.where(
                    next_mask > 0.5,
                    q_next_online,
                    torch.tensor(float('-inf'), device=self.device)
                )
                next_a = torch.argmax(q_next_online_masked, dim=1, keepdim=True)
                
                q_next_target = self.q_target(s_next, next_feats)
                max_next = q_next_target.gather(1, next_a).squeeze(1)
                
                action_was_valid = next_mask.gather(1, next_a).squeeze(1) > 0.5
                max_next = torch.where(action_was_valid, max_next, torch.zeros_like(max_next))
                
            else:
                q_next = self.q_target(s_next, next_feats)
                q_next_masked = torch.where(
                    next_mask > 0.5,
                    q_next,
                    torch.tensor(float('-inf'), device=self.device)
                )
                max_next = torch.max(q_next_masked, dim=1)[0]

            target_all = r + (1.0 - done) * self.cfg.gamma * max_next

        target = target_all[valid]

        loss = F.smooth_l1_loss(q_sa, target)
        
        self.opt.zero_grad()
        loss.backward()
        
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        
        self.opt.step()
        
        self.training_steps += 1

        if self.training_steps % self.cfg.target_update_interval == 0:
            self._update_target_network()
        
        return loss.item()

    def _update_target_network(self):
        if self.cfg.tau > 0:
            for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
                )
        else:
            self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({
            'q_state_dict': self.q.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'env_steps': self.env_steps,
            'training_steps': self.training_steps,
            'epsilon': self._eps,
            'config': self.cfg,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint['q_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.env_steps = checkpoint['env_steps']
        self.training_steps = checkpoint['training_steps']
        self._eps = checkpoint['epsilon']