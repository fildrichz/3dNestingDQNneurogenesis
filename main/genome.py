"""
Genome representation for GA-based neural architecture evolution.

Based on "Using Genetic Algorithms to Optimize Artificial Neural Networks"
(Ding et al., 2010), Section 3.2: Optimizing Network Architecture.

The genome encodes network architecture parameters using parameterization coding,
which captures the most important characteristics: number of layers, neurons,
connections, and structural components.
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
    
    Learning parameters (Section 3.3) are kept fixed for this study.
    """
    
    # Define search spaces for each gene (Section 3.2 parameters only)
    GENE_SPACES = {
        # Core architecture
        'hidden_dim': [128, 192, 256, 320, 384, 512],      # Number of neurons
        'enc_layers': [1, 2, 3, 4],                         # Number of encoder layers
        'head_hidden': [128, 192, 256, 320, 384],          # Head layer size

        # Attention mechanism
        'attention_type': ['standard', 'set_transformer', 'none'],  # Attention type
        'attention_heads': [2, 4, 8],                      # Number of attention heads
        'num_inducing_points': [16, 32, 64],               # Inducing points for set_transformer

        # Spatial feature extraction
        'patch_size': [5, 7, 9],                           # Heightmap patch size
        'cnn_channels': [[8, 16], [16, 32], [32, 64], [16, 48]],  # CNN channel progression

        # Regularization
        'dropout': [0.0, 0.05, 0.1, 0.15, 0.2],           # Dropout rate

        # Activation function
        'activation': ['relu', 'gelu', 'silu'],            # Activation function type
    }
    
    # Fixed parameters (not evolved)
    FIXED_PARAMS = {
        'lr': 1e-4,
        'batch_size': 128,
        'gamma': 0.992,
        'n_step': 3,
        'eps_start': 1.0,
        'eps_end': 0.15,
        'eps_decay_steps': 20_000,
        'target_update_interval': 500,
        'double_dqn': True,
        'warmup_steps': 1000,
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
            # Use random.choice for nested lists (like cnn_channels)
            # np.random.choice doesn't work with 2D lists
            value = random.choice(space)
            # Convert to native Python types immediately
            genes[gene_name] = cls._convert_to_python(value)
        return genes

    def to_dqn_config(self, obs_dim: int, action_feat_dim: int,
                     max_actions: int, device: str = "cpu") -> DQNConfigEnhanced:
        """
        Decode genome into DQN configuration.

        This converts the genotype (gene representation) to phenotype
        (actual network configuration).

        Args:
            obs_dim: Observation dimension
            action_feat_dim: Action feature dimension
            max_actions: Maximum number of actions
            device: Device to run on

        Returns:
            DQNConfigEnhanced instance
        """
        # Determine if attention should be used
        attention_type = self.genes['attention_type']
        use_attention = attention_type != 'none'

        return DQNConfigEnhanced(
            obs_dim=obs_dim,
            action_feat_dim=action_feat_dim,
            max_actions=max_actions,
            device=device,

            # EVOLVED PARAMETERS (Section 3.2)
            hidden=self.genes['hidden_dim'],
            enc_layers=self.genes['enc_layers'],
            head_hidden=self.genes['head_hidden'],
            use_attention=use_attention,
            attention_type=attention_type if use_attention else 'standard',
            attention_heads=self.genes['attention_heads'],  # ✅ Now passed to config
            num_inducing_points=self.genes['num_inducing_points'],
            heightmap_patch_size=self.genes['patch_size'],
            cnn_channels=self.genes['cnn_channels'],
            dropout=self.genes['dropout'],
            activation=self.genes['activation'],

            # FIXED PARAMETERS (not part of Section 3.2 evolution)
            **self.FIXED_PARAMS
        )
    

    def mutate(self, mutation_rate: float = 0.2) -> 'NetworkGenome':
        """Create mutated copy of genome."""
        import random
        new_genes = self.genes.copy()

        for gene_name, space in self.GENE_SPACES.items():
            if np.random.rand() < mutation_rate:
                # Use random.choice for nested lists (like cnn_channels)
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
