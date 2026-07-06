"""
Genetic-algorithm neurogenesis for the RELATIONAL DQN architecture.

Same evolutionary scheme as ga_evolution.py (adaptive population sizing,
elitism, tournament selection, uniform/single-point crossover, mutation-rate
decay), but genomes describe the relational constraint-learning network
(genome_relational.RelationalGenome) and are evaluated by round-robin
training + greedy evaluation via
packing_with_relational.train_relational_multi_problem.
Fitness = 0.80*utilization + 0.20*bins-efficiency (no parsimony pressure:
utilization differences are small in absolute value but significant, and a
complexity penalty overly punishes larger networks).

Because the relational network is problem-agnostic (item IDs never enter
it), a genome can be evaluated on SEVERAL problems at once and the evolved
architecture directly transfers to unseen problems.
"""
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from genome_relational import RelationalGenome, create_initial_relational_population
from ga_evolution import get_adaptive_population_size, tournament_selection


def evaluate_relational_genome(genome: RelationalGenome,
                               train_paths: List[str],
                               episodes: int = 60,
                               device: str = None,
                               seed: int = 42,
                               train_freq: int = 2,
                               verbose: bool = False) -> Tuple[float, Dict]:
    """Fitness = 0.80*avg_greedy_util + 0.20*bins_efficiency (no parsimony)."""
    import torch
    from packing_with_relational import train_relational_multi_problem

    overrides = genome.to_cfg_overrides()
    violation_penalty = overrides.pop('violation_penalty')
    # eps genes are trainer-managed; RelationalDQNConfig fields eps_start/eps_end
    # are set through overrides directly (they are config fields)

    agent = None
    try:
        agent, results = train_relational_multi_problem(
            train_paths=train_paths,
            test_paths=None,
            episodes=episodes,
            seed=seed,
            device=device,
            violation_penalty=violation_penalty,
            train_freq=train_freq,
            cfg_overrides=overrides,
            verbose=verbose,
        )
        summary = results['summary'].get('train', {})
    finally:
        if agent is not None:
            del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_util = summary.get('avg_utilization', 0.0)
    avg_bins = summary.get('avg_bins_used', 0.0)
    complexity = genome.get_network_complexity()

    # NOTE: no parsimony term - utilization differences between architectures
    # are small in absolute value but significant for the objective, and a
    # complexity penalty was found to overly punish larger networks.
    # Complexity is still reported in metrics for analysis.
    bins_penalty = min(avg_bins / 5.0, 1.0)
    fitness = 0.80 * avg_util + 0.20 * (1.0 - bins_penalty)

    genome.fitness = fitness
    genome.metrics = dict(summary)
    genome.metrics['complexity'] = complexity
    genome.metrics['fitness_components'] = {
        'utilization': avg_util,
        'bins_efficiency': 1.0 - bins_penalty,
    }
    return fitness, genome.metrics


def evolve_relational_architecture(
    train_paths: List[str],
    generations: int = 10,
    population_size: int = 12,
    episodes_per_eval: int = 60,
    elite_size: int = 2,
    mutation_rate: float = 0.2,
    adaptive_mutation: bool = True,
    adaptive_population: bool = True,
    exploration_ratio: float = 1.25,
    exploitation_ratio: float = 0.75,
    crossover_method: str = 'uniform',
    tournament_size: int = 3,
    seed: Optional[int] = None,
    device: str = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[RelationalGenome, List[RelationalGenome]]:
    """Main GA loop for relational architecture evolution."""
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print("GA NEUROGENESIS - RELATIONAL (constraint-learning) ARCHITECTURE")
        print(f"{'='*80}")
        print(f"Training problems ({len(train_paths)}): "
              f"{[p.split('/')[-1] for p in train_paths]}")
        print(f"Base population: {population_size} | generations: {generations} | "
              f"episodes/eval: {episodes_per_eval}")
        print(f"{'='*80}\n")

    init_size = (get_adaptive_population_size(0, generations, population_size,
                                              exploration_ratio, exploitation_ratio)
                 if adaptive_population else population_size)
    population = create_initial_relational_population(init_size)
    next_genome_id = init_size

    fitness_cache: Dict[str, Tuple[float, Dict]] = {}
    best_ever: Optional[RelationalGenome] = None
    history = []

    for gen in range(generations):
        t_gen = time.time()
        current_rate = (mutation_rate * (1.0 - 0.5 * gen / max(1, generations - 1))
                        if adaptive_mutation else mutation_rate)

        if verbose:
            print(f"\n{'='*80}\nGENERATION {gen+1}/{generations} "
                  f"(population {len(population)}, mutation {current_rate:.3f})\n{'='*80}")

        for i, genome in enumerate(population):
            key = genome.get_cache_key({'episodes': episodes_per_eval})
            if key in fitness_cache:
                genome.fitness, genome.metrics = fitness_cache[key]
                cached = " (cached)"
            else:
                evaluate_relational_genome(
                    genome, train_paths, episodes=episodes_per_eval,
                    device=device, seed=42)
                fitness_cache[key] = (genome.fitness, genome.metrics)
                cached = ""
            if verbose:
                print(f"[{i+1}/{len(population)}] genome {genome.genome_id}: "
                      f"fitness={genome.fitness:.4f}{cached} | "
                      f"util={genome.metrics.get('avg_utilization', 0):.3f} | "
                      f"bins={genome.metrics.get('avg_bins_used', 0):.1f} | "
                      f"viol={genome.metrics.get('avg_greedy_violations', 0):.1f} | "
                      f"{genome.get_network_complexity():.2f}M params")

        population.sort(key=lambda g: g.fitness, reverse=True)
        gen_best = population[0]
        if best_ever is None or gen_best.fitness > best_ever.fitness:
            best_ever = gen_best
            if verbose:
                print(f"\n*** NEW BEST GENOME (fitness {best_ever.fitness:.4f}):")
                print(f"    {best_ever}")

        fits = [g.fitness for g in population]
        history.append({
            'generation': gen + 1,
            'population_size': len(population),
            'best_fitness': float(np.max(fits)),
            'avg_fitness': float(np.mean(fits)),
            'std_fitness': float(np.std(fits)),
            'best_utilization': gen_best.metrics.get('avg_utilization', 0.0),
            'best_greedy_violations': gen_best.metrics.get('avg_greedy_violations', 0.0),
            'mutation_rate': current_rate,
            'best_genome': gen_best.to_dict(),
            'elapsed_s': time.time() - t_gen,
        })
        if save_dir:
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            with open(save_dir / "best_genome.json", "w") as f:
                json.dump(best_ever.to_dict(), f, indent=2)

        if verbose:
            print(f"\nGen {gen+1} summary: best={np.max(fits):.4f} "
                  f"avg={np.mean(fits):.4f}+-{np.std(fits):.4f} "
                  f"({time.time()-t_gen:.0f}s)")

        if gen == generations - 1:
            break

        # ---- next generation ------------------------------------------------
        next_size = (get_adaptive_population_size(gen + 1, generations, population_size,
                                                  exploration_ratio, exploitation_ratio)
                     if adaptive_population else population_size)
        next_pop = [copy_genome(g) for g in population[:elite_size]]
        while len(next_pop) < next_size:
            p1 = tournament_selection(population, tournament_size)
            p2 = tournament_selection(population, tournament_size)
            child = RelationalGenome.crossover(p1, p2, method=crossover_method)
            child = child.mutate(current_rate)
            child.genome_id = next_genome_id
            next_genome_id += 1
            next_pop.append(child)
        population = next_pop

    if verbose:
        print(f"\n{'='*80}\nEVOLUTION COMPLETE\nBest genome:\n{best_ever}\n{'='*80}")
    return best_ever, population


def copy_genome(g: RelationalGenome) -> RelationalGenome:
    """Elite copy that keeps fitness/metrics (avoids re-evaluation via cache)."""
    import copy as _copy
    c = RelationalGenome(_copy.deepcopy(g.genes), genome_id=g.genome_id)
    c.fitness, c.metrics = g.fitness, _copy.deepcopy(g.metrics)
    return c


if __name__ == "__main__":
    from packing_with_relational import full_datapath

    best, pop = evolve_relational_architecture(
        train_paths=[full_datapath("3dBPP_11.txt"), full_datapath("3dBPP_12.txt")],
        generations=6,
        population_size=10,
        episodes_per_eval=60,
        save_dir="output_data/ga_relational",
    )
