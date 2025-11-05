"""
Parameterization-Based Neurogenesis for Fully-Connected DQN

This implements the PARAMETERIZATION CODING approach from Section 3.2 of the paper,
which is ideal for fully-connected networks where we only need to evolve:
- Number of layers
- Neurons per layer  
- Activation functions

Paper: "Parameterization coding only codes the most important characteristics 
of the related architecture, such as the number of hidden layers, the number 
of nodes of each layer..."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import copy
import random

# Import base components

#can you see other files?
#a: Yes


from dqn_core.dqn_base import DQNConfig, DQNAgent, to_torch


# ============================================================================
# PARAMETERIZATION GENOME (Simplified for Fully-Connected Networks)
# ============================================================================

@dataclass
class FCGenome:
    """
    Parameterization-based genome for fully-connected networks.
    
    Paper Section 3.2: "Parameterization coding only codes the most important
    characteristics of the related architecture..."
    
    For fully-connected DQN, the key parameters are:
    - state_encoder_layers: neurons per layer in state encoder
    - action_encoder_layers: neurons per layer in action encoder
    - head_layers: neurons per layer in Q-value head
    - activation: activation function
    """
    state_encoder_layers: List[int] = field(default_factory=lambda: [256, 128])
    action_encoder_layers: List[int] = field(default_factory=lambda: [256, 128])
    head_layers: List[int] = field(default_factory=lambda: [128])
    activation: str = 'relu'
    
    # Evolution metadata
    fitness: float = 0.0
    age: int = 0
    
    @staticmethod
    def from_base_dqn(hidden: int = 256, enc_layers: int = 2, head_hidden: int = 256) -> FCGenome:
        """
        Create genome that recreates the DQN_base architecture.
        
        This ensures you can reproduce your current architecture exactly.
        
        Args:
            hidden: Hidden dimension (from DQN_base)
            enc_layers: Number of encoder layers (from DQN_base)
            head_hidden: Head hidden dimension (from DQN_base)
            
        Returns:
            FCGenome matching DQN_base architecture
        """
        # DQN_base uses same layers for state and action encoder
        encoder_layers = [hidden] * max(1, enc_layers - 1) + [hidden]
        
        return FCGenome(
            state_encoder_layers=encoder_layers.copy(),
            action_encoder_layers=encoder_layers.copy(),
            head_layers=[head_hidden],
            activation='relu'
        )
    
    @staticmethod
    def random_genome(min_layers: int = 1, 
                     max_layers: int = 4,
                     min_neurons: int = 32,
                     max_neurons: int = 512) -> FCGenome:
        """
        Create a random genome within specified bounds.
        
        Args:
            min_layers: Minimum layers per encoder
            max_layers: Maximum layers per encoder
            min_neurons: Minimum neurons per layer
            max_neurons: Maximum neurons per layer
        """
        # Random number of layers
        state_depth = random.randint(min_layers, max_layers)
        action_depth = random.randint(min_layers, max_layers)
        head_depth = random.randint(1, 2)
        
        # Random neuron counts (typically decreasing)
        state_layers = []
        for i in range(state_depth):
            # Geometric decrease: start larger, end smaller
            neurons = random.choice([32, 64, 128, 256, 512])
            neurons = min(max_neurons, max(min_neurons, neurons))
            state_layers.append(neurons)
        
        action_layers = []
        for i in range(action_depth):
            neurons = random.choice([32, 64, 128, 256, 512])
            neurons = min(max_neurons, max(min_neurons, neurons))
            action_layers.append(neurons)
        
        head_layers = []
        for i in range(head_depth):
            neurons = random.choice([32, 64, 128, 256])
            neurons = min(max_neurons // 2, max(min_neurons, neurons))
            head_layers.append(neurons)
        
        # Random activation
        activation = random.choice(['relu', 'tanh', 'elu', 'leaky_relu', 'gelu'])
        
        return FCGenome(
            state_encoder_layers=state_layers,
            action_encoder_layers=action_layers,
            head_layers=head_layers,
            activation=activation
        )
    
    def copy(self) -> FCGenome:
        """Deep copy of genome"""
        return FCGenome(
            state_encoder_layers=self.state_encoder_layers.copy(),
            action_encoder_layers=self.action_encoder_layers.copy(),
            head_layers=self.head_layers.copy(),
            activation=self.activation,
            fitness=self.fitness,
            age=self.age
        )
    
    def mutate(self, mutation_rate: float = 0.15) -> None:
        """
        Mutate genome parameters.
        
        Paper mentions: "basic bit mutation operator or well-proportioned mutation operator"
        
        Mutation types for fully-connected networks:
        1. Add/remove layer
        2. Change neuron count in layer
        3. Change activation function
        """
        # Mutate state encoder
        if random.random() < mutation_rate:
            self._mutate_layer_list(self.state_encoder_layers)
        
        # Mutate action encoder
        if random.random() < mutation_rate:
            self._mutate_layer_list(self.action_encoder_layers)
        
        # Mutate head
        if random.random() < mutation_rate:
            self._mutate_layer_list(self.head_layers)
        
        # Mutate activation
        if random.random() < mutation_rate / 2:  # Less frequent
            activations = ['relu', 'tanh', 'elu', 'leaky_relu', 'gelu', 'sigmoid']
            self.activation = random.choice(activations)
    
    def _mutate_layer_list(self, layers: List[int]) -> None:
        """Mutate a list of layer sizes"""
        mutation_type = random.choice(['add', 'remove', 'change', 'change'])
        
        if mutation_type == 'add' and len(layers) < 5:
            # Add a new layer
            pos = random.randint(0, len(layers))
            new_size = random.choice([32, 64, 128, 256, 512])
            layers.insert(pos, new_size)
        
        elif mutation_type == 'remove' and len(layers) > 1:
            # Remove a layer
            pos = random.randint(0, len(layers) - 1)
            layers.pop(pos)
        
        elif mutation_type == 'change' and layers:
            # Change size of existing layer
            pos = random.randint(0, len(layers) - 1)
            current = layers[pos]
            
            # Small change: ±25-50%
            if random.random() < 0.5:
                change = random.choice([0.5, 0.75, 1.5, 2.0])
                new_size = int(current * change)
            else:
                # Jump to nearby power of 2
                new_size = random.choice([32, 64, 128, 256, 512])
            
            layers[pos] = max(8, min(1024, new_size))
    
    def to_config(self, base_config: DQNConfig) -> DQNConfig:
        """
        Convert genome to DQNConfig.
        
        This creates a config that can be used with your existing DQN code.
        """
        config = copy.deepcopy(base_config)
        
        # Set architecture from genome
        # Note: DQN_base uses 'hidden' and 'enc_layers', we need to set reasonable values
        # The actual architecture will be built using these layer lists
        config.hidden = self.state_encoder_layers[0] if self.state_encoder_layers else 256
        config.enc_layers = len(self.state_encoder_layers)
        config.head_hidden = self.head_layers[0] if self.head_layers else 128
        
        # Store full architecture (for evolvable networks)
        if not hasattr(config, 'state_encoder_layers'):
            # Add these attributes for use with EvolvableQNetwork
            config.state_encoder_layers = self.state_encoder_layers
            config.action_encoder_layers = self.action_encoder_layers
            config.head_layers = self.head_layers
            config.activation = self.activation
        else:
            config.state_encoder_layers = self.state_encoder_layers
            config.action_encoder_layers = self.action_encoder_layers
            config.head_layers = self.head_layers
            config.activation = self.activation
        
        return config
    
    def complexity_penalty(self, weight: float = 0.0001) -> float:
        """
        Calculate complexity penalty to prevent overgrown networks.
        
        Paper: "individual fitness can be defined by the error between the 
        expected output and actual output and the network complexity"
        
        Args:
            weight: Penalty weight (tune this based on your task)
        """
        total_neurons = (
            sum(self.state_encoder_layers) +
            sum(self.action_encoder_layers) +
            sum(self.head_layers)
        )
        
        # Also penalize too many layers
        total_layers = (
            len(self.state_encoder_layers) +
            len(self.action_encoder_layers) +
            len(self.head_layers)
        )
        
        return weight * (total_neurons + 10 * total_layers)
    
    def get_param_count(self, obs_dim: int, action_feat_dim: int) -> int:
        """
        Estimate total parameter count.
        
        Useful for understanding model size.
        """
        # State encoder
        params = 0
        prev = obs_dim
        for size in self.state_encoder_layers:
            params += prev * size + size  # weights + bias
            prev = size
        state_out = prev
        
        # Action encoder
        prev = action_feat_dim
        for size in self.action_encoder_layers:
            params += prev * size + size
            prev = size
        action_out = prev
        
        # Head
        prev = state_out + action_out
        for size in self.head_layers:
            params += prev * size + size
            prev = size
        params += prev * 1 + 1  # Final output layer
        
        return params
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return (
            f"FCGenome(\n"
            f"  state: {self.state_encoder_layers}\n"
            f"  action: {self.action_encoder_layers}\n"
            f"  head: {self.head_layers}\n"
            f"  activation: {self.activation}\n"
            f"  fitness: {self.fitness:.4f}\n"
            f")"
        )


# ============================================================================
# GENETIC OPERATORS (Same as before, but cleaner)
# ============================================================================

def crossover_fc(parent1: FCGenome, parent2: FCGenome) -> Tuple[FCGenome, FCGenome]:
    """
    Crossover for parameterization genomes.
    
    Paper: "single point crossover operator"
    
    For fully-connected networks, we swap entire encoder/head configurations.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Randomly swap components
    if random.random() < 0.5:
        child1.state_encoder_layers, child2.state_encoder_layers = \
            child2.state_encoder_layers.copy(), child1.state_encoder_layers.copy()
    
    if random.random() < 0.5:
        child1.action_encoder_layers, child2.action_encoder_layers = \
            child2.action_encoder_layers.copy(), child1.action_encoder_layers.copy()
    
    if random.random() < 0.5:
        child1.head_layers, child2.head_layers = \
            child2.head_layers.copy(), child1.head_layers.copy()
    
    if random.random() < 0.5:
        child1.activation, child2.activation = child2.activation, child1.activation
    
    child1.fitness = 0.0
    child2.fitness = 0.0
    child1.age = 0
    child2.age = 0
    
    return child1, child2


def tournament_selection_fc(population: List[FCGenome], 
                            tournament_size: int = 3) -> FCGenome:
    """Tournament selection"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda g: g.fitness)


# ============================================================================
# POPULATION MANAGER FOR FC NETWORKS
# ============================================================================

class FCPopulationManager:
    """
    Population manager specialized for fully-connected network evolution.
    
    Simpler than the general PopulationManager since we only deal with
    parameterization encoding.
    """
    
    def __init__(self,
                 population_size: int = 20,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        self.population: List[FCGenome] = []
        self.generation = 0
        self.best_genome: Optional[FCGenome] = None
        self.best_fitness = float('-inf')
        
        # Statistics
        self.fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.complexity_history: List[float] = []
    
    def initialize_population(self, 
                            base_genome: Optional[FCGenome] = None,
                            min_layers: int = 1,
                            max_layers: int = 4) -> None:
        """
        Initialize population.
        
        Args:
            base_genome: Optional starting genome (e.g., your DQN_base config)
            min_layers: Minimum layers for random genomes
            max_layers: Maximum layers for random genomes
        """
        self.population = []
        
        # Optionally seed with base genome
        if base_genome is not None:
            self.population.append(base_genome.copy())
        
        # Fill rest with random genomes
        while len(self.population) < self.population_size:
            genome = FCGenome.random_genome(
                min_layers=min_layers,
                max_layers=max_layers
            )
            self.population.append(genome)
        
        self.generation = 0
    
    def evolve_generation(self) -> None:
        """
        Evolve population by one generation.
        
        Implements the GA loop from Figure 2 in the paper.
        """
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Update best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_genome = self.population[0].copy()
        
        # Statistics
        fitnesses = [g.fitness for g in self.population]
        self.fitness_history.append(max(fitnesses))
        self.mean_fitness_history.append(np.mean(fitnesses))
        
        complexities = [g.complexity_penalty() for g in self.population]
        self.complexity_history.append(np.mean(complexities))
        
        # Elitism: preserve best
        new_population = [g.copy() for g in self.population[:self.elitism_count]]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = tournament_selection_fc(self.population)
            parent2 = tournament_selection_fc(self.population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = crossover_fc(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to size
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Age genomes
        for genome in self.population:
            genome.age += 1
    
    def get_best_genome(self) -> FCGenome:
        """Get best genome from current population"""
        return max(self.population, key=lambda g: g.fitness)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics"""
        fitnesses = [g.fitness for g in self.population]
        complexities = [g.complexity_penalty() for g in self.population]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'mean_complexity': np.mean(complexities),
            'population_size': len(self.population),
            'best_ever_fitness': self.best_fitness
        }


# ============================================================================
# CONTROLLER FOR FC NETWORKS
# ============================================================================

class FCNeurogenesisController:
    """
    Simplified controller for fully-connected network evolution.
    
    This is easier to use than the general controller for FC networks.
    """
    
    def __init__(self,
                 base_config: DQNConfig,
                 population_size: int = 20,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 2):
        self.base_config = base_config
        self.pop_manager = FCPopulationManager(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism_count=elitism_count
        )
        self.device = torch.device(base_config.device)
    
    def initialize_from_base(self) -> None:
        """
        Initialize population seeded with your DQN_base architecture.
        
        This ensures the evolution starts from a known good baseline.
        """
        # Create genome matching DQN_base
        base_genome = FCGenome.from_base_dqn(
            hidden=self.base_config.hidden,
            enc_layers=self.base_config.enc_layers,
            head_hidden=self.base_config.head_hidden
        )
        
        self.pop_manager.initialize_population(base_genome=base_genome)
    
    def initialize_random(self, min_layers: int = 1, max_layers: int = 4) -> None:
        """Initialize with random genomes"""
        self.pop_manager.initialize_population(
            base_genome=None,
            min_layers=min_layers,
            max_layers=max_layers
        )
    
    def build_agent_from_genome(self, genome: FCGenome) -> DQNAgent:
        """
        Build a DQN agent from a genome.
        
        This returns a standard DQNAgent that works with your existing code.
        """
        config = genome.to_config(self.base_config)
        
        # Use EvolvableDQNAgent if available, otherwise standard
        try:
            from dqn_evolvable import EvolvableDQNAgent, EvolvableDQNConfig
            # Convert to evolvable config
            evolve_config = EvolvableDQNConfig(**vars(config))
            return EvolvableDQNAgent(evolve_config)
        except ImportError:
            # Fall back to standard agent
            return DQNAgent(config)
    
    def evolve(self,
               generations: int,
               evaluate_fn,
               verbose: bool = True) -> FCGenome:
        """
        Run evolution for specified generations.
        
        Args:
            generations: Number of generations
            evaluate_fn: Function that takes (genome, agent) and returns fitness
            verbose: Print progress
            
        Returns:
            Best genome found
        """
        for gen in range(generations):
            if verbose:
                print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate population
            for i, genome in enumerate(self.pop_manager.population):
                agent = self.build_agent_from_genome(genome)
                fitness = evaluate_fn(genome, agent)
                genome.fitness = fitness
                
                if verbose and i % 1 == 0:
                    print(f"  Evaluated {i+1}/{len(self.pop_manager.population)}", end='\r')
            
            # Print statistics
            if verbose:
                stats = self.pop_manager.get_statistics()
                print(f"\n  Best: {stats['best_fitness']:.4f}")
                print(f"  Mean: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
                print(f"  Complexity: {stats['mean_complexity']:.6f}")
            
            # Evolve to next generation
            if gen < generations - 1:
                self.pop_manager.evolve_generation()
        
        return self.pop_manager.get_best_genome()
    
    def get_best_genome(self) -> FCGenome:
        """Get best genome found"""
        return self.pop_manager.best_genome or self.pop_manager.get_best_genome()
    
    def save_best_genome(self, path: str) -> None:
        """Save best genome"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.get_best_genome(), f)
    
    def load_genome(self, path: str) -> FCGenome:
        """Load genome"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FC Neurogenesis Example")
    print("=" * 60)
    
    # 1. Create base config (your DQN_base settings)
    base_config = DQNConfig(
        obs_dim=10,
        action_feat_dim=5,
        max_actions=4,
        hidden=256,
        enc_layers=2,
        head_hidden=256,
        device='cpu'
    )
    
    # 2. Verify we can recreate DQN_base architecture
    print("\n1. Recreating DQN_base architecture:")
    base_genome = FCGenome.from_base_dqn(
        hidden=base_config.hidden,
        enc_layers=base_config.enc_layers,
        head_hidden=base_config.head_hidden
    )
    print(base_genome)
    print(f"   Parameters: {base_genome.get_param_count(10, 5):,}")
    
    # 3. Create controller
    print("\n2. Initializing controller:")
    controller = FCNeurogenesisController(
        base_config=base_config,
        population_size=10,
        mutation_rate=0.15
    )
    
    # 4. Initialize from base (ensures we start from known good architecture)
    controller.initialize_from_base()
    print(f"   Population initialized with {len(controller.pop_manager.population)} genomes")
    print(f"   First genome (base): {controller.pop_manager.population[0]}")
    
    # 5. Example evolution (with dummy evaluation)
    print("\n3. Running evolution (dummy evaluation):")
    
    def dummy_evaluate(genome: FCGenome, agent: DQNAgent) -> float:
        """Dummy evaluation - replace with real training"""
        # Prefer moderate-sized networks
        param_count = genome.get_param_count(10, 5)
        target_size = 50000  # Target ~50k parameters
        
        # Fitness: penalize deviation from target
        size_penalty = abs(param_count - target_size) / target_size
        fitness = 1.0 - size_penalty - genome.complexity_penalty()
        return fitness
    
    best_genome = controller.evolve(
        generations=3,
        evaluate_fn=dummy_evaluate,
        verbose=True
    )
    
    print("\n4. Best genome found:")
    print(best_genome)
    print(f"   Parameters: {best_genome.get_param_count(10, 5):,}")
    
    # 6. Build agent from best genome
    print("\n5. Building agent from best genome:")
    agent = controller.build_agent_from_genome(best_genome)
    print(f"   Agent created: {type(agent).__name__}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
