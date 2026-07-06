"""
Genome for GA-based evolution of the RELATIONAL DQN architecture
(dqn_core/dqn_relational.py), mirroring the NetworkGenome API in genome.py
so the same GA machinery (tournament selection, adaptive population sizing)
applies.

Evolved genes cover:
- the relational encoder (d_model, n_layers, n_heads, ffn_mult, dropout)
- the compositional Q head (comp_dim)
- spatial state description (bin heightmap grid, EMS patch size, CNN channels)
- training hyperparameters (lr, batch_size, gamma, n_step)
- the constraint-violation penalty itself (violation_penalty) - since
  constraints are learned from refusals rather than masked, the right
  penalty magnitude is a hyperparameter worth evolving.
"""
import copy
from typing import Dict, Any, List

import numpy as np


class RelationalGenome:
    """Evolvable genome for the relational (constraint-learning) DQN."""

    # every listed d_model is divisible by every listed n_heads
    GENE_SPACES = {
        # relational encoder
        'd_model': [64, 96, 128, 192, 256],
        'n_layers': [1, 2, 3, 4],
        'n_heads': [2, 4, 8],
        'ffn_mult': [2, 4],
        'dropout': [0.0, 0.05, 0.1],

        # compositional Q head
        'comp_dim': [32, 64, 128],

        # spatial state description
        'grid': [16, 24, 32],
        'patch_size': [5, 7, 9],
        'cnn_channels': [[8, 16], [16, 32], [32, 64]],

        # training hyperparameters
        'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'gamma': [0.98, 0.985, 0.99, 0.992, 0.995],
        'n_step': [1, 3, 5],

        # environment referee
        'violation_penalty': [0.0, 0.05, 0.1, 0.25],
    }

    # not evolved (kept small for GA-phase evaluations)
    FIXED_PARAMS = {
        'buffer_size': 15_000,
        'warmup_steps': 300,
        'target_update_interval': 200,
        'double_dqn': True,
        'grad_clip': 1.0,
        'eps_start': 1.0,
        'eps_end': 0.01,
    }

    def __init__(self, genes: Dict[str, Any] = None, genome_id: int = None):
        self.genes = genes if genes is not None else self.random_genes()
        self.genome_id = genome_id
        self.fitness = None
        self.metrics = {}

    @classmethod
    def random_genes(cls) -> Dict[str, Any]:
        import random
        return {name: copy.deepcopy(random.choice(space))
                for name, space in cls.GENE_SPACES.items()}

    def to_cfg_overrides(self) -> Dict[str, Any]:
        """Config overrides for RelationalDQNConfig via cfg_overrides.
        The violation_penalty gene is NOT a config field - pop it and pass
        it to the environment/trainer separately."""
        g = self.genes
        d, heads = int(g['d_model']), int(g['n_heads'])
        if d % heads != 0:  # defensive: pick the largest valid divisor
            heads = max(h for h in self.GENE_SPACES['n_heads'] if d % h == 0)
        overrides = {
            'd_model': d,
            'n_layers': int(g['n_layers']),
            'n_heads': heads,
            'ffn_mult': int(g['ffn_mult']),
            'dropout': float(g['dropout']),
            'comp_dim': int(g['comp_dim']),
            'grid': int(g['grid']),
            'patch_size': int(g['patch_size']),
            'cnn_channels': tuple(g['cnn_channels']),
            'lr': float(g['lr']),
            'batch_size': int(g['batch_size']),
            'gamma': float(g['gamma']),
            'n_step': int(g['n_step']),
            'violation_penalty': float(g['violation_penalty']),
        }
        overrides.update(self.FIXED_PARAMS)
        return overrides

    def mutate(self, mutation_rate: float = 0.2) -> 'RelationalGenome':
        import random
        new_genes = copy.deepcopy(self.genes)
        for name, space in self.GENE_SPACES.items():
            if np.random.rand() < mutation_rate:
                new_genes[name] = copy.deepcopy(random.choice(space))
        return RelationalGenome(new_genes)

    @classmethod
    def crossover(cls, p1: 'RelationalGenome', p2: 'RelationalGenome',
                  method: str = 'uniform') -> 'RelationalGenome':
        names = list(cls.GENE_SPACES.keys())
        child = {}
        if method == 'uniform':
            for n in names:
                src = p1 if np.random.rand() < 0.5 else p2
                child[n] = copy.deepcopy(src.genes[n])
        elif method == 'single_point':
            point = np.random.randint(1, len(names))
            for i, n in enumerate(names):
                src = p1 if i < point else p2
                child[n] = copy.deepcopy(src.genes[n])
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        return RelationalGenome(child)

    def get_network_complexity(self) -> float:
        """Rough parameter count in millions (for parsimony pressure)."""
        d = self.genes['d_model']
        dc = self.genes['comp_dim']
        f = self.genes['ffn_mult']
        L = self.genes['n_layers']
        c0, c1 = self.genes['cnn_channels']

        per_layer = (4 + 2 * f) * d * d          # qkv+out + ffn
        encoder = L * per_layer
        inputs = (10 + 7 + 12 + 4) * d           # token input projections
        cnns = 2 * (c0 * 9 + c0 * c1 * 9 + c1 * d)  # bin-heightmap + EMS-patch CNNs
        head = 2 * (2 * d * d + d * dc)          # u/v projections
        return (encoder + inputs + cnns + head) / 1e6

    def get_cache_key(self, extra: Dict[str, Any] = None) -> str:
        import json
        genes = dict(sorted(self.genes.items()))
        key = json.dumps(genes, sort_keys=True)
        if extra:
            key += "|" + json.dumps(dict(sorted(extra.items())), sort_keys=True)
        return key

    def to_dict(self) -> Dict[str, Any]:
        return {'genome_id': self.genome_id,
                'genes': copy.deepcopy(self.genes),
                'fitness': self.fitness,
                'metrics': copy.deepcopy(self.metrics),
                'complexity': self.get_network_complexity()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationalGenome':
        g = cls(genes=data['genes'], genome_id=data.get('genome_id'))
        g.fitness = data.get('fitness')
        g.metrics = data.get('metrics', {})
        return g

    def __repr__(self) -> str:
        gene_str = ', '.join(f"{k}={v}" for k, v in self.genes.items())
        fit = f", fitness={self.fitness:.3f}" if self.fitness is not None else ""
        return f"RelationalGenome({gene_str}{fit})"


def create_initial_relational_population(size: int) -> List[RelationalGenome]:
    return [RelationalGenome(genome_id=i) for i in range(size)]
