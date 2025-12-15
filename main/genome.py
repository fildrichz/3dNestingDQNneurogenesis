"""
Genome representation for GA-based neural architecture evolution.

Based on "Using Genetic Algorithms to Optimize Artificial Neural Networks"
(Ding et al., 2010), Section 3.2: Optimizing Network Architecture.

The genome encodes network architecture parameters using parameterization coding,
which captures the most important characteristics: number of layers, neurons,
connections, and structural components.

EXTENDED: Co-evolution of architecture and training hyperparameters (Neuvo NAS+ 2025).
Now also evolves learning rate, batch size, and discount factor alongside architecture.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dqn_core.dqn_enhanced import DQNConfigEnhanced


class NetworkGenome:
    """
    Encodes neural network architecture as an evolvable genome.

    Genes represent architectural parameters (Section 3.2):
    - Number of hidden layers
    - Number of neurons per layer
    - Attention mechanism configuration
    - Spatial feature extraction parameters

    NEW: Also evolves key training hyperparameters (Neuvo NAS+ 2025):
    - Learning rate, batch size, discount factor
    - Other learning parameters (Section 3.3) remain fixed
    """
    
    # Define search spaces for each gene (Section 3.2 parameters + hyperparameters)
    # Multiplicative genes: {'type': 'multiplicative', 'base': X, 'min': Y, 'max': Z}
    # Discrete genes: list of valid values
    GENE_SPACES = {
        # Core architecture - multiplicative scaling (unbounded search)
        'hidden_dim': {'type': 'multiplicative', 'base': 32, 'min': 8, 'max': 512},
        'enc_layers': {'type': 'multiplicative', 'base': 2, 'min': 1, 'max': 8},
        'head_hidden': {'type': 'multiplicative', 'base': 32, 'min': 16, 'max': 256},

        # Attention mechanism
        'attention_type': ['standard', 'set_transformer', 'none'],  # Discrete
        'attention_heads': {'type': 'multiplicative', 'base': 4, 'min': 2, 'max': 16},
        'num_inducing_points': {'type': 'multiplicative', 'base': 32, 'min': 8, 'max': 128},

        # Spatial feature extraction
        'patch_size': {'type': 'multiplicative', 'base': 7, 'min': 3, 'max': 15},
        'cnn_channels': [[8, 16], [16, 32], [32, 64], [16, 48]],  # Discrete (complex structure)

        # Regularization - discrete (non-integer floats)
        'dropout': [0.0, 0.05, 0.1, 0.15, 0.2],

        # Activation function - discrete (categorical)
        'activation': ['relu', 'gelu', 'silu'],

        # Training hyperparameters (NEW: Neuvo NAS+ 2025 - co-evolution of architecture & hyperparameters)
        'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'batch_size': [64, 128, 256],
        'gamma': [0.98, 0.985, 0.99, 0.992, 0.995],
    }
    
    # Fixed parameters (not evolved)
    # NOTE: lr, batch_size, gamma are now evolved as genes (Neuvo NAS+ 2025)
    # NOTE: eps_decay_episodes and total_episodes are calculated automatically in to_dqn_config()
    FIXED_PARAMS = {
        'n_step': 3,  # Reduced from 15: Better for short training episodes during GA
        'eps_start': 1.0,
        'eps_end': 0.01,  # Reduced from 0.15 to 0.01 (1% minimum exploration)
        # eps_decay_episodes and total_episodes are NOT in FIXED_PARAMS
        # They are set dynamically by calculate_dynamic_epsilon_decay() in training functions
        'target_update_interval': 200,  # Reduced from 500 for faster target updates
        'double_dqn': True,
        'warmup_steps': 300,  # Reduced from 1000: Start training earlier
        'buffer_size': 200_000,
        'grad_clip': 1.0,
    }
    
    def __init__(self, genes: Dict[str, Any] = None, genome_id: int = None):
        """
        Initialize genome with genes.
        
        Args:
            genes: Dictionary of gene values. If None, randomly initialize.
            genome_id: Optional identifier for tracking
        """
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        
        self.genome_id = genome_id
        self.fitness = None
        self.metrics = {}
    
    @classmethod
    def random_genes(cls) -> Dict[str, Any]:
        """Generate random genome with valid gene values."""
        import random
        genes = {}
        for gene_name, space in cls.GENE_SPACES.items():
            if isinstance(space, dict) and space.get('type') == 'multiplicative':
                # Multiplicative gene: randomly initialize within bounds
                # Generate valid powers of 2 within the range
                min_val, max_val = space['min'], space['max']

                # Find all valid powers of 2 in range
                valid_values = []
                current = min_val
                while current <= max_val:
                    valid_values.append(current)
                    current *= 2

                # If we don't have any valid powers of 2, use min/max bounds
                if not valid_values:
                    valid_values = [min_val, max_val]

                # Randomly select from valid values
                genes[gene_name] = random.choice(valid_values)
            else:
                # Discrete gene: random choice from list
                value = random.choice(space)
                # Convert to native Python types immediately
                genes[gene_name] = cls._convert_to_python(value)
        return genes

    def to_dqn_config(self, obs_dim: int, action_feat_dim: int,
                     max_actions: int, device: str = "cpu",
                     total_episodes: int = None) -> DQNConfigEnhanced:
        # Calculate epsilon decay if total_episodes provided
        if total_episodes is not None:
            from dqn_core.dqn_enhanced import calculate_dynamic_epsilon_decay
            eps_decay_episodes, total_eps = calculate_dynamic_epsilon_decay(
                total_episodes=total_episodes,
                plateau_at_ratio=0.8,
                verbose=False
            )
        else:
            eps_decay_episodes = None
            total_eps = None

        # Determine if attention should be used
        attention_type = self.genes['attention_type']
        use_attention = attention_type != 'none'

        hidden = int(self.genes['hidden_dim'])
        requested_heads = int(self.genes['attention_heads'])

        # --- FIX: enforce heads divides hidden (only relevant when attention is used) ---
        if use_attention:
            # Allowed heads from the gene space (works for your multiplicative powers-of-2 space)
            space = self.GENE_SPACES['attention_heads']
            allowed_heads = []
            if isinstance(space, dict) and space.get('type') == 'multiplicative':
                h = int(space['min'])
                while h <= int(space['max']):
                    allowed_heads.append(h)
                    h *= 2
            else:
                allowed_heads = [int(x) for x in space]

            # Choose the largest allowed head count that is <= requested_heads and divides hidden
            valid = [h for h in allowed_heads if h <= requested_heads and h <= hidden and (hidden % h == 0)]
            if not valid:
                # Fallback: choose any allowed divisor of hidden
                valid = [h for h in allowed_heads if h <= hidden and (hidden % h == 0)]
            safe_heads = max(valid) if valid else 1
        else:
            safe_heads = 1
        # --- END FIX ---

        return DQNConfigEnhanced(
            obs_dim=obs_dim,
            action_feat_dim=action_feat_dim,
            max_actions=max_actions,
            device=device,

            hidden=hidden,
            enc_layers=self.genes['enc_layers'],
            head_hidden=self.genes['head_hidden'],
            use_attention=use_attention,
            attention_type=attention_type if use_attention else 'standard',
            attention_heads=safe_heads,
            num_inducing_points=self.genes['num_inducing_points'],
            heightmap_patch_size=self.genes['patch_size'],
            cnn_channels=self.genes['cnn_channels'],
            dropout=self.genes['dropout'],
            activation=self.genes['activation'],

            lr=self.genes['lr'],
            batch_size=self.genes['batch_size'],
            gamma=self.genes['gamma'],

            eps_decay_episodes=eps_decay_episodes,
            total_episodes=total_eps,

            **self.FIXED_PARAMS
        )

    

    def mutate(self, mutation_rate: float = 0.2) -> 'NetworkGenome':
        """
        Create mutated copy of genome.

        For multiplicative genes: randomly multiply or divide by 2, clamped to bounds
        For discrete genes: randomly select from available options
        """
        import random
        new_genes = self.genes.copy()

        for gene_name, space in self.GENE_SPACES.items():
            if np.random.rand() < mutation_rate:
                if isinstance(space, dict) and space.get('type') == 'multiplicative':
                    # Multiplicative mutation: randomly *2 or /2
                    current_value = new_genes[gene_name]
                    if np.random.rand() < 0.5:
                        # Multiply by 2
                        new_value = current_value * 2
                    else:
                        # Divide by 2
                        new_value = current_value // 2  # Integer division

                    # Clamp to bounds
                    new_value = max(space['min'], min(space['max'], new_value))
                    new_genes[gene_name] = int(new_value)
                else:
                    # Discrete mutation: random choice from list
                    value = random.choice(space)
                    new_genes[gene_name] = self._convert_to_python(value)

        return NetworkGenome(new_genes)
    
    @staticmethod
    def _convert_to_python(value):
        """Convert numpy types to native Python types."""
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (list, np.ndarray)):
            # Handle list-type genes (e.g., cnn_channels)
            return [int(v) if isinstance(v, np.integer) else v for v in value]
        return value
    
    @classmethod
    def crossover(cls, parent1: 'NetworkGenome', parent2: 'NetworkGenome',
                 method: str = 'uniform') -> 'NetworkGenome':
        """
        Create child genome through crossover of two parents.

        Supports:
        - 'uniform': Each gene randomly chosen from either parent
        - 'single_point': Single crossover point

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            method: Crossover method ('uniform' or 'single_point')

        Returns:
            New child NetworkGenome
        """
        import copy
        gene_names = list(cls.GENE_SPACES.keys())
        child_genes = {}

        if method == 'uniform':
            # Uniform crossover: randomly choose each gene from either parent
            for gene_name in gene_names:
                if np.random.rand() < 0.5:
                    # Deep copy to avoid aliasing issues with list genes
                    child_genes[gene_name] = copy.deepcopy(parent1.genes[gene_name])
                else:
                    child_genes[gene_name] = copy.deepcopy(parent2.genes[gene_name])

        elif method == 'single_point':
            # Single-point crossover
            crossover_point = np.random.randint(1, len(gene_names))

            for i, gene_name in enumerate(gene_names):
                if i < crossover_point:
                    child_genes[gene_name] = copy.deepcopy(parent1.genes[gene_name])
                else:
                    child_genes[gene_name] = copy.deepcopy(parent2.genes[gene_name])

        else:
            raise ValueError(f"Unknown crossover method: {method}")

        return NetworkGenome(child_genes)
    
    def get_network_complexity(self) -> float:
        """
        Estimate network complexity for parsimony pressure.

        Complexity is roughly proportional to number of parameters.
        Used in fitness calculation to prefer smaller networks.

        Returns:
            Complexity score (higher = more complex)
        """
        hidden = self.genes['hidden_dim']
        layers = self.genes['enc_layers']
        head_hidden = self.genes['head_hidden']
        attention_type = self.genes['attention_type']

        complexity = 0

        # State encoder MLP (obs_dim=8 -> hidden, with 'layers' hidden layers)
        # Layer 1: 8 -> hidden
        complexity += 8 * hidden
        # Intermediate layers: hidden -> hidden
        for _ in range(max(0, layers - 1)):
            complexity += hidden * hidden

        # Action encoder MLP (action_feat_dim+64 -> hidden, with 'layers' hidden layers)
        # Layer 1: (25+64)=89 -> hidden
        complexity += 89 * hidden
        # Intermediate layers: hidden -> hidden
        for _ in range(max(0, layers - 1)):
            complexity += hidden * hidden

        # CNN parameters
        cnn_channels = self.genes['cnn_channels']
        # Conv1: 1 -> cnn_channels[0], kernel 3x3
        complexity += 1 * cnn_channels[0] * 9
        # Conv2: cnn_channels[0] -> cnn_channels[1], kernel 3x3
        complexity += cnn_channels[0] * cnn_channels[1] * 9
        # FC: cnn_channels[1] -> 64
        complexity += cnn_channels[1] * 64

        # Attention parameters
        if attention_type == 'standard':
            # TransformerEncoder with 2 layers
            # Each layer: Q,K,V projections + output proj + FFN
            for _ in range(2):
                # Multi-head attention (4 projections: Q, K, V, O)
                complexity += 4 * (hidden * hidden)
                # FFN: hidden -> 2*hidden -> hidden
                complexity += hidden * (2 * hidden) + (2 * hidden) * hidden

        elif attention_type == 'set_transformer':
            num_inds = self.genes['num_inducing_points']
            # 2 ISAB blocks, each with 2 MAB modules
            # Simplified estimate: 4 MAB modules total
            for _ in range(4):
                # Q, K, V, O projections
                complexity += 4 * (hidden * hidden)
                # Inducing points
                complexity += num_inds * hidden

        # Head: (2*hidden) -> head_hidden -> 1
        complexity += (2 * hidden) * head_hidden
        complexity += head_hidden * 1

        return complexity / 1e6  # Normalize to millions of parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Export genome as dictionary for logging/serialization."""
        return {
            'genome_id': self.genome_id,
            'genes': self.genes.copy(),
            'fitness': self.fitness,
            'metrics': self.metrics.copy(),
            'complexity': self.get_network_complexity()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkGenome':
        """Load genome from dictionary."""
        genome = cls(genes=data['genes'], genome_id=data.get('genome_id'))
        genome.fitness = data.get('fitness')
        genome.metrics = data.get('metrics', {})
        return genome

    def get_cache_key(self, curriculum_params: Dict[str, Any] = None) -> str:
        """
        Generate a unique cache key for this genome configuration.

        Args:
            curriculum_params: Optional dict with curriculum settings (item_fraction, episodes)
                              to differentiate evaluations at different difficulty levels

        Returns:
            String key that uniquely identifies this genome + curriculum combination
        """
        import json

        # Sort genes to ensure consistent ordering
        genes_sorted = dict(sorted(self.genes.items()))

        # Convert to canonical JSON string (handles lists consistently)
        genes_str = json.dumps(genes_sorted, sort_keys=True)

        # Add curriculum parameters if provided
        if curriculum_params:
            curriculum_sorted = dict(sorted(curriculum_params.items()))
            curriculum_str = json.dumps(curriculum_sorted, sort_keys=True)
            return f"{genes_str}|{curriculum_str}"

        return genes_str

    def __repr__(self) -> str:
        """String representation of genome."""
        gene_str = ', '.join(f"{k}={v}" for k, v in self.genes.items())
        fitness_str = f", fitness={self.fitness:.3f}" if self.fitness is not None else ""
        return f"NetworkGenome({gene_str}{fitness_str})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        lines = ["Network Architecture Genome:"]
        lines.append(f"  ID: {self.genome_id}")
        lines.append("  Genes:")
        for gene_name, value in self.genes.items():
            lines.append(f"    {gene_name}: {value}")
        if self.fitness is not None:
            lines.append(f"  Fitness: {self.fitness:.4f}")
        if self.metrics:
            lines.append("  Metrics:")
            for metric, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {metric}: {value:.4f}")
                else:
                    lines.append(f"    {metric}: {value}")
        return '\n'.join(lines)


def create_initial_population(population_size: int) -> List[NetworkGenome]:
    """
    Create initial population of random genomes.
    
    Args:
        population_size: Number of genomes to create
        
    Returns:
        List of NetworkGenome instances
    """
    return [NetworkGenome(genome_id=i) for i in range(population_size)]


def tournament_selection(population: List[NetworkGenome], 
                        tournament_size: int = 3) -> NetworkGenome:
    """
    Select genome using tournament selection.
    
    Randomly select tournament_size genomes and return the best one.
    
    Args:
        population: List of genomes with fitness scores
        tournament_size: Number of genomes in tournament
        
    Returns:
        Selected genome
    """
    tournament = np.random.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda g: g.fitness if g.fitness is not None else -np.inf)