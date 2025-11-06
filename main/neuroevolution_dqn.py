"""
Neuroevolution for 3D Bin Packing DQN
=====================================
Implements GA-based architecture evolution using indirect coding (parameterization).

Inspired by: "Using Genetic Algorithms to Optimize Artificial Neural Networks" 
(Ding et al., 2010)

Key Idea:
- Genotype = DQNConfig parameters (hidden_dim, layers, etc.)
- Population = Multiple DQN architectures
- Fitness = Packing performance (utilization, bins used, etc.)
- Evolution = Selection + Crossover + Mutation
- Training = Gradient descent optimizes weights for each architecture
"""

import numpy as np
import torch
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, replace
import json
from collections import deque

from dqn_core.dqn_base import DQNAgent, DQNConfig


@dataclass
class ArchitectureGene:
    """
    Genotype: Encodes neural architecture using indirect coding (parameterization).
    
    Each gene represents a DQN architecture configuration.
    Only encodes the most important architectural characteristics.
    """
    # Encoder architecture
    hidden_dim: int          # {64, 128, 256, 512}
    enc_layers: int          # {1, 2, 3, 4}
    
    # Head architecture  
    head_hidden: int         # {64, 128, 256, 512}
    
    # Learning parameters (can also evolve!)
    lr: float                # {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}
    gamma: float             # {0.97, 0.98, 0.985, 0.99, 0.995}
    
    # Optional: Advanced features
    n_step: int              # {1, 3, 5}
    batch_size: int          # {32, 64, 128, 256}
    
    # Fitness tracking
    fitness: float = 0.0
    episodes_trained: int = 0
    
    def to_config(self, base_config: DQNConfig) -> DQNConfig:
        """Convert gene to DQNConfig"""
        return replace(
            base_config,
            hidden=self.hidden_dim,
            enc_layers=self.enc_layers,
            head_hidden=self.head_hidden,
            lr=self.lr,
            gamma=self.gamma,
            n_step=self.n_step,
            batch_size=self.batch_size
        )
    
    @staticmethod
    def from_config(config: DQNConfig) -> 'ArchitectureGene':
        """Create gene from DQNConfig"""
        return ArchitectureGene(
            hidden_dim=config.hidden,
            enc_layers=config.enc_layers,
            head_hidden=config.head_hidden,
            lr=config.lr,
            gamma=config.gamma,
            n_step=config.n_step,
            batch_size=config.batch_size
        )
    
    def clone(self) -> 'ArchitectureGene':
        """Deep copy"""
        return copy.deepcopy(self)
    
    def complexity(self) -> int:
        """Estimate parameter count (for regularization)"""
        # Rough approximation
        return (self.hidden_dim ** 2) * self.enc_layers + self.head_hidden * self.hidden_dim


@dataclass
class Individual:
    """
    Individual in the population.
    Contains: genotype (architecture), phenotype (agent), and fitness.
    """
    gene: ArchitectureGene
    agent: Optional[DQNAgent] = None
    
    # Performance metrics (used for fitness)
    avg_utilization: float = 0.0
    avg_bins_used: float = 0.0
    avg_items_placed: float = 0.0
    episodes_evaluated: int = 0
    
    def compute_fitness(self, 
                       complexity_penalty: float = 0.0001,
                       bins_weight: float = 0.3,
                       util_weight: float = 0.5,
                       items_weight: float = 0.2) -> float:
        """
        Multi-objective fitness function:
        - Maximize utilization
        - Minimize bins used (normalized by max_bins)
        - Maximize items placed (normalized by total items)
        - Minimize model complexity (optional)
        """
        if self.episodes_evaluated == 0:
            return 0.0
        
        # Normalize bins (lower is better, so invert)
        bins_score = 1.0 - (self.avg_bins_used / 10.0)  # Assume max 10 bins
        
        # Utilization and items are already normalized [0, 1]
        performance = (
            util_weight * self.avg_utilization +
            bins_weight * bins_score +
            items_weight * self.avg_items_placed
        )
        
        # Complexity penalty (encourage smaller models)
        complexity_cost = complexity_penalty * self.gene.complexity()
        
        fitness = performance - complexity_cost
        self.gene.fitness = fitness
        return fitness


class NeuroEvolutionDQN:
    """
    Genetic Algorithm for evolving DQN architectures.
    
    Process:
    1. Initialize population of N architectures
    2. Train each architecture for M episodes
    3. Evaluate fitness based on performance
    4. Select top-k individuals (elitism)
    5. Generate offspring via crossover + mutation
    6. Repeat
    
    Hybrid approach (as paper recommends):
    - GA searches architecture space (global)
    - Gradient descent trains weights (local)
    """
    
    def __init__(self,
                 base_config: DQNConfig,
                 population_size: int = 10,
                 elite_size: int = 3,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 architecture_space: Optional[Dict] = None):
        """
        Args:
            base_config: Base DQN configuration (provides fixed params)
            population_size: Number of architectures in population
            elite_size: Number of top performers to keep unchanged
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover vs. cloning
            architecture_space: Valid values for each parameter
        """
        self.base_config = base_config
        self.pop_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define search space for each architectural parameter
        self.arch_space = architecture_space or {
            'hidden_dim': [64, 128, 256, 512],
            'enc_layers': [1, 2, 3, 4],
            'head_hidden': [64, 128, 256, 512],
            'lr': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            'gamma': [0.97, 0.98, 0.985, 0.99, 0.992],
            'n_step': [1, 3, 5],
            'batch_size': [32, 64, 128, 256]
        }
        
        # Initialize population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history = []
        
        print(f" NeuroEvolution initialized:")
        print(f"   Population size: {self.pop_size}")
        print(f"   Elite size: {self.elite_size}")
        print(f"   Mutation rate: {self.mutation_rate}")
        print(f"   Crossover rate: {self.crossover_rate}")
    
    def initialize_population(self) -> None:
        """Step 1: Randomly generate N architectures"""
        print(f"\n Initializing population of {self.pop_size} architectures...")
        
        self.population = []
        for i in range(self.pop_size):
            # Random architecture from search space
            gene = ArchitectureGene(
                hidden_dim=np.random.choice(self.arch_space['hidden_dim']),
                enc_layers=np.random.choice(self.arch_space['enc_layers']),
                head_hidden=np.random.choice(self.arch_space['head_hidden']),
                lr=np.random.choice(self.arch_space['lr']),
                gamma=np.random.choice(self.arch_space['gamma']),
                n_step=np.random.choice(self.arch_space['n_step']),
                batch_size=np.random.choice(self.arch_space['batch_size'])
            )
            
            # Create agent with this architecture
            config = gene.to_config(self.base_config)
            agent = DQNAgent(config)
            
            individual = Individual(gene=gene, agent=agent)
            self.population.append(individual)
            
            print(f"   Individual {i+1}: h={gene.hidden_dim}, layers={gene.enc_layers}, "
                  f"head={gene.head_hidden}, lr={gene.lr:.0e}")
    
    def evaluate_individual(self, 
                          individual: Individual,
                          env,
                          num_episodes: int,
                          items: List,
                          build_features_fn,
                          pad_feats_mask_fn,
                          max_actions: int,
                          action_feat_dim: int,
                          train_freq: int = 1,
                          verbose: bool = False) -> Dict:
        """
        Step 2: Train architecture for M episodes and collect performance.
        
        This is the "hybrid" approach:
        - GA optimizes architecture
        - Gradient descent optimizes weights
        """
        agent = individual.agent
        
        metrics = {
            'utilizations': [],
            'bins_used': [],
            'items_placed': [],
            'returns': []
        }
        
        for ep in range(num_episodes):
            obs = env.reset(items=items.copy())
            ep_return = 0.0
            steps = 0
            
            while True:
                # Get actions
                actions, mask_short = env.action_space()
                feats = build_features_fn(env, actions) if len(actions) > 0 else \
                        np.zeros((0, action_feat_dim), np.float32)
                
                # Select action
                act_idx = agent.select_action(
                    obs,
                    feats if feats.shape[0] > 0 else np.zeros((1, action_feat_dim), np.float32),
                    mask_short if mask_short.shape[0] > 0 else np.zeros((1,), np.float32)
                )
                act = None if (act_idx is None or actions == [] or 
                             actions[act_idx] is None) else actions[act_idx]
                
                # Prepare current features
                currF, currM = pad_feats_mask_fn(
                    feats if feats.shape[0] > 0 else np.zeros((0, action_feat_dim), np.float32),
                    mask_short if mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                    max_actions
                )
                
                # Step environment
                nobs, rew, done, info = env.step(act)
                
                # Prepare next features
                n_actions, n_mask_short = env.action_space()
                n_feats = build_features_fn(env, n_actions) if len(n_actions) > 0 else \
                         np.zeros((0, action_feat_dim), np.float32)
                nextF, nextM = pad_feats_mask_fn(
                    n_feats,
                    n_mask_short if n_mask_short.shape[0] > 0 else np.zeros((0,), np.float32),
                    max_actions
                )
                
                # Store transition
                agent.store(obs, act_idx, rew, nobs, done,
                          curr_action_feats=currF, curr_mask=currM,
                          next_action_feats=nextF, next_mask=nextM)
                
                # Train
                if steps % train_freq == 0:
                    agent.train_step()
                
                obs = nobs
                ep_return += rew
                steps += 1
                
                if done:
                    # Record metrics
                    metrics['utilizations'].append(info.get('utilization', 0.0))
                    metrics['bins_used'].append(info.get('bins_used', 0))
                    metrics['items_placed'].append(info.get('items_placed', 0) / len(items))
                    metrics['returns'].append(ep_return)
                    
                    if verbose and (ep + 1) % 10 == 0:
                        print(f"      Ep {ep+1}/{num_episodes}: "
                              f"Util={info.get('utilization', 0):.3f}, "
                              f"Bins={info.get('bins_used', 0)}, "
                              f"Items={info.get('items_placed', 0)}/{len(items)}")
                    break
        
        # Update individual metrics
        individual.avg_utilization = np.mean(metrics['utilizations'])
        individual.avg_bins_used = np.mean(metrics['bins_used'])
        individual.avg_items_placed = np.mean(metrics['items_placed'])
        individual.episodes_evaluated += num_episodes
        individual.gene.episodes_trained += num_episodes
        
        return metrics
    
    def selection(self) -> List[Individual]:
        """
        Step 4: Select top individuals based on fitness (elitism).
        """
        # Compute fitness for all
        for ind in self.population:
            ind.compute_fitness()
        
        # Sort by fitness (descending)
        sorted_pop = sorted(self.population, key=lambda x: x.gene.fitness, reverse=True)
        
        # Update best individual
        if self.best_individual is None or sorted_pop[0].gene.fitness > self.best_individual.gene.fitness:
            self.best_individual = copy.deepcopy(sorted_pop[0])
        
        # Keep elite
        elite = sorted_pop[:self.elite_size]
        
        # Tournament selection for the rest
        mating_pool = list(elite)  # Elites always included
        tournament_size = 3
        
        while len(mating_pool) < self.pop_size:
            # Random tournament
            contestants = np.random.choice(sorted_pop, size=tournament_size, replace=False)
            winner = max(contestants, key=lambda x: x.gene.fitness)
            mating_pool.append(winner)
        
        return mating_pool
    
    def crossover(self, parent1: ArchitectureGene, parent2: ArchitectureGene) -> ArchitectureGene:
        """
        Step 5: Uniform crossover - randomly inherit each parameter from parents.
        """
        child = ArchitectureGene(
            hidden_dim=np.random.choice([parent1.hidden_dim, parent2.hidden_dim]),
            enc_layers=np.random.choice([parent1.enc_layers, parent2.enc_layers]),
            head_hidden=np.random.choice([parent1.head_hidden, parent2.head_hidden]),
            lr=np.random.choice([parent1.lr, parent2.lr]),
            gamma=np.random.choice([parent1.gamma, parent2.gamma]),
            n_step=np.random.choice([parent1.n_step, parent2.n_step]),
            batch_size=np.random.choice([parent1.batch_size, parent2.batch_size])
        )
        return child
    
    def mutate(self, gene: ArchitectureGene) -> ArchitectureGene:
        """
        Step 5: Mutation - randomly change parameters with mutation_rate probability.
        """
        mutated = gene.clone()
        
        if np.random.rand() < self.mutation_rate:
            mutated.hidden_dim = np.random.choice(self.arch_space['hidden_dim'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.enc_layers = np.random.choice(self.arch_space['enc_layers'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.head_hidden = np.random.choice(self.arch_space['head_hidden'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.lr = np.random.choice(self.arch_space['lr'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.gamma = np.random.choice(self.arch_space['gamma'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.n_step = np.random.choice(self.arch_space['n_step'])
        
        if np.random.rand() < self.mutation_rate:
            mutated.batch_size = np.random.choice(self.arch_space['batch_size'])
        
        return mutated
    
    def reproduce(self, mating_pool: List[Individual]) -> List[Individual]:
        """
        Step 5: Generate offspring via crossover and mutation.
        Elites pass through unchanged.
        """
        offspring = []
        
        # Elites go through unchanged
        for i in range(self.elite_size):
            elite_copy = Individual(
                gene=mating_pool[i].gene.clone(),
                agent=None  # Will create fresh agent
            )
            offspring.append(elite_copy)
        
        # Generate rest via crossover + mutation
        while len(offspring) < self.pop_size:
            # Select two parents randomly
            parent1, parent2 = np.random.choice(mating_pool, size=2, replace=False)
            
            # Crossover or clone
            if np.random.rand() < self.crossover_rate:
                child_gene = self.crossover(parent1.gene, parent2.gene)
            else:
                child_gene = parent1.gene.clone()
            
            # Mutate
            child_gene = self.mutate(child_gene)
            
            offspring.append(Individual(gene=child_gene, agent=None))
        
        return offspring
    
    def evolve_generation(self,
                         env,
                         items: List,
                         episodes_per_individual: int,
                         build_features_fn,
                         pad_feats_mask_fn,
                         max_actions: int,
                         action_feat_dim: int,
                         train_freq: int = 1) -> Dict:
        """
        Complete evolution cycle for one generation.
        
        Returns statistics about the generation.
        """
        print(f"\n{'='*80}")
        print(f" GENERATION {self.generation + 1}")
        print(f"{'='*80}")
        
        # Step 2: Train and evaluate each architecture
        print(f"\n Evaluating {len(self.population)} architectures ({episodes_per_individual} episodes each)...")
        
        for i, individual in enumerate(self.population):
            # Create fresh agent if needed
            if individual.agent is None:
                config = individual.gene.to_config(self.base_config)
                individual.agent = DQNAgent(config)
            
            print(f"\n   [{i+1}/{len(self.population)}] Training architecture: "
                  f"h={individual.gene.hidden_dim}, layers={individual.gene.enc_layers}, "
                  f"head={individual.gene.head_hidden}")
            
            self.evaluate_individual(
                individual, env, episodes_per_individual, items,
                build_features_fn, pad_feats_mask_fn,
                max_actions, action_feat_dim, train_freq,
                verbose=(i < 3)  # Show details for first 3
            )
        
        # Step 3: Compute fitness
        for ind in self.population:
            ind.compute_fitness()
        
        # Statistics
        fitnesses = [ind.gene.fitness for ind in self.population]
        utils = [ind.avg_utilization for ind in self.population]
        bins = [ind.avg_bins_used for ind in self.population]
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_util': max(utils),
            'mean_util': np.mean(utils),
            'best_bins': min(bins),
            'mean_bins': np.mean(bins)
        }
        
        self.fitness_history.append(stats)
        
        # Step 4: Selection
        mating_pool = self.selection()
        
        print(f"\n Generation {self.generation + 1} Results:")
        print(f"   Best fitness: {stats['best_fitness']:.4f}")
        print(f"   Mean fitness: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
        print(f"   Best util: {stats['best_util']:.3f} | Mean util: {stats['mean_util']:.3f}")
        print(f"   Best bins: {stats['best_bins']:.1f} | Mean bins: {stats['mean_bins']:.1f}")
        
        # Show best architecture
        best_ind = max(self.population, key=lambda x: x.gene.fitness)
        print(f"\n   Best Architecture:")
        print(f"      Hidden: {best_ind.gene.hidden_dim}")
        print(f"      Enc layers: {best_ind.gene.enc_layers}")
        print(f"      Head hidden: {best_ind.gene.head_hidden}")
        print(f"      LR: {best_ind.gene.lr:.0e}")
        print(f"      Gamma: {best_ind.gene.gamma}")
        print(f"      N-step: {best_ind.gene.n_step}")
        print(f"      Batch size: {best_ind.gene.batch_size}")
        print(f"      Complexity: {best_ind.gene.complexity():,} params (approx)")
        
        # Step 5: Reproduction
        self.population = self.reproduce(mating_pool)
        self.generation += 1
        
        return stats
    
    def save_best(self, path: str) -> None:
        """Save best architecture and agent"""
        if self.best_individual is None:
            print(" No best individual to save yet")
            return
        
        # Save agent weights
        self.best_individual.agent.save(path)
        
        # Save architecture config (convert numpy types to Python natives)
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            config_dict = {
                'hidden_dim': int(self.best_individual.gene.hidden_dim),
                'enc_layers': int(self.best_individual.gene.enc_layers),
                'head_hidden': int(self.best_individual.gene.head_hidden),
                'lr': float(self.best_individual.gene.lr),
                'gamma': float(self.best_individual.gene.gamma),
                'n_step': int(self.best_individual.gene.n_step),
                'batch_size': int(self.best_individual.gene.batch_size),
                'fitness': float(self.best_individual.gene.fitness),
                'avg_utilization': float(self.best_individual.avg_utilization),
                'avg_bins_used': float(self.best_individual.avg_bins_used)
            }
            json.dump(config_dict, f, indent=2)
        
        print(f"Best architecture saved to {path} and {config_path}")
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """Visualize evolution progress"""
        import matplotlib.pyplot as plt
        
        if not self.fitness_history:
            print(" No evolution history to plot")
            return
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        mean_fitness = [h['mean_fitness'] for h in self.fitness_history]
        best_util = [h['best_util'] for h in self.fitness_history]
        mean_util = [h['mean_util'] for h in self.fitness_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Fitness evolution
        axes[0].plot(generations, best_fitness, 'g-', linewidth=2, label='Best')
        axes[0].plot(generations, mean_fitness, 'b--', label='Mean')
        axes[0].fill_between(generations, 
                            [h['mean_fitness'] - h['std_fitness'] for h in self.fitness_history],
                            [h['mean_fitness'] + h['std_fitness'] for h in self.fitness_history],
                            alpha=0.2, color='blue')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Utilization evolution
        axes[1].plot(generations, best_util, 'g-', linewidth=2, label='Best')
        axes[1].plot(generations, mean_util, 'b--', label='Mean')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Utilization')
        axes[1].set_title('Utilization Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Evolution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()