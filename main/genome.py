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
        'use_attention': [True, False],                     # Enable/disable attention
        'attention_heads': [2, 4, 8],                      # Number of attention heads
        
        # Spatial feature extraction
        'patch_size': [5, 7, 9],                           # Heightmap patch size
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
        genes = {}
        for gene_name, space in cls.GENE_SPACES.items():
            value = np.random.choice(space)
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
        return DQNConfigEnhanced(
            obs_dim=obs_dim,
            action_feat_dim=action_feat_dim,
            max_actions=max_actions,
            device=device,
            
            # EVOLVED PARAMETERS (Section 3.2)
            hidden=self.genes['hidden_dim'],
            enc_layers=self.genes['enc_layers'],
            head_hidden=self.genes['head_hidden'],
            use_attention=self.genes['use_attention'],
            heightmap_patch_size=self.genes['patch_size'],
            
            # FIXED PARAMETERS (not part of Section 3.2 evolution)
            **self.FIXED_PARAMS
        )
    

    def mutate(self, mutation_rate: float = 0.2) -> 'NetworkGenome':
        """Create mutated copy of genome."""
        new_genes = self.genes.copy()
        
        for gene_name, space in self.GENE_SPACES.items():
            if np.random.rand() < mutation_rate:
                value = np.random.choice(space)
                new_genes[gene_name] = self._convert_to_python(value)
        
        return NetworkGenome(new_genes)
    
    @staticmethod
    def _convert_to_python(value):
        """Convert numpy types to native Python types."""
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.bool_):
            return bool(value)
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
        gene_names = list(cls.GENE_SPACES.keys())
        child_genes = {}
        
        if method == 'uniform':
            # Uniform crossover: randomly choose each gene from either parent
            for gene_name in gene_names:
                if np.random.rand() < 0.5:
                    child_genes[gene_name] = parent1.genes[gene_name]
                else:
                    child_genes[gene_name] = parent2.genes[gene_name]
        
        elif method == 'single_point':
            # Single-point crossover
            crossover_point = np.random.randint(1, len(gene_names))
            
            for i, gene_name in enumerate(gene_names):
                if i < crossover_point:
                    child_genes[gene_name] = parent1.genes[gene_name]
                else:
                    child_genes[gene_name] = parent2.genes[gene_name]
        
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
        # Rough parameter count estimate
        hidden = self.genes['hidden_dim']
        layers = self.genes['enc_layers']
        head = self.genes['head_hidden']
        attention = self.genes['use_attention']
        
        # Base MLP parameters
        complexity = hidden * layers * 2  # Encoder parameters
        complexity += head  # Head parameters
        
        # Attention adds significant parameters
        if attention:
            heads = self.genes['attention_heads']
            complexity += hidden * hidden * heads * 2  # Attention parameters
        
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
