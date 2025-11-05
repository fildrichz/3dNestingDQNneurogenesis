"""
Neuroevolution Analysis & Visualization
=======================================

Tools for analyzing and visualizing neuroevolution results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns


def load_evolution_history(config_paths: List[str]) -> List[Dict]:
    """Load evolution history from saved configs"""
    history = []
    for path in sorted(config_paths):
        with open(path, 'r') as f:
            history.append(json.load(f))
    return history


def plot_comprehensive_evolution(neuro_evo, save_path: str = None):
    """
    Create comprehensive visualization of evolution progress.
    """
    if not neuro_evo.fitness_history:
        print("No evolution history to visualize")
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Fitness Evolution
    ax1 = plt.subplot(2, 3, 1)
    generations = [h['generation'] for h in neuro_evo.fitness_history]
    best_fitness = [h['best_fitness'] for h in neuro_evo.fitness_history]
    mean_fitness = [h['mean_fitness'] for h in neuro_evo.fitness_history]
    std_fitness = [h['std_fitness'] for h in neuro_evo.fitness_history]
    
    ax1.plot(generations, best_fitness, 'g-', linewidth=2.5, marker='o', label='Best', markersize=6)
    ax1.plot(generations, mean_fitness, 'b--', linewidth=2, marker='s', label='Mean', markersize=5)
    ax1.fill_between(generations, 
                     np.array(mean_fitness) - np.array(std_fitness),
                     np.array(mean_fitness) + np.array(std_fitness),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness', fontsize=11)
    ax1.set_title('Fitness Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Utilization Evolution
    ax2 = plt.subplot(2, 3, 2)
    best_util = [h['best_util'] for h in neuro_evo.fitness_history]
    mean_util = [h['mean_util'] for h in neuro_evo.fitness_history]
    
    ax2.plot(generations, best_util, 'g-', linewidth=2.5, marker='o', label='Best', markersize=6)
    ax2.plot(generations, mean_util, 'b--', linewidth=2, marker='s', label='Mean', markersize=5)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Utilization', fontsize=11)
    ax2.set_title('Utilization Evolution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Bins Used Evolution
    ax3 = plt.subplot(2, 3, 3)
    best_bins = [h['best_bins'] for h in neuro_evo.fitness_history]
    mean_bins = [h['mean_bins'] for h in neuro_evo.fitness_history]
    
    ax3.plot(generations, best_bins, 'g-', linewidth=2.5, marker='o', label='Best (fewer is better)', markersize=6)
    ax3.plot(generations, mean_bins, 'b--', linewidth=2, marker='s', label='Mean', markersize=5)
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Bins Used', fontsize=11)
    ax3.set_title('Bins Used Evolution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Lower is better
    
    # 4. Diversity (Std Dev Evolution)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(generations, std_fitness, 'r-', linewidth=2, marker='D', markersize=5)
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Fitness Std Dev', fontsize=11)
    ax4.set_title('Population Diversity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, 0.95, 'Lower = More Converged', 
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Improvement Rate
    ax5 = plt.subplot(2, 3, 5)
    if len(best_fitness) > 1:
        improvements = [best_fitness[i] - best_fitness[i-1] 
                       for i in range(1, len(best_fitness))]
        ax5.bar(generations[1:], improvements, color='green', alpha=0.7, edgecolor='black')
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax5.set_xlabel('Generation', fontsize=11)
        ax5.set_ylabel('Fitness Improvement', fontsize=11)
        ax5.set_title('Per-Generation Improvement', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Architecture Distribution (Final Generation)
    ax6 = plt.subplot(2, 3, 6)
    if neuro_evo.population:
        # Collect architecture parameters
        hidden_dims = [ind.gene.hidden_dim for ind in neuro_evo.population]
        enc_layers = [ind.gene.enc_layers for ind in neuro_evo.population]
        
        # Create histogram
        param_counts = {}
        for h, l in zip(hidden_dims, enc_layers):
            key = f"h={h},l={l}"
            param_counts[key] = param_counts.get(key, 0) + 1
        
        labels = list(param_counts.keys())
        values = list(param_counts.values())
        
        ax6.bar(range(len(labels)), values, color='skyblue', edgecolor='black', alpha=0.7)
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Architecture Distribution (Final Gen)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Comprehensive evolution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_architecture_comparison(individuals: List, save_path: str = None):
    """
    Compare different architectures in the population.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect data
    hidden_dims = [ind.gene.hidden_dim for ind in individuals]
    enc_layers = [ind.gene.enc_layers for ind in individuals]
    head_hidden = [ind.gene.head_hidden for ind in individuals]
    learning_rates = [ind.gene.lr for ind in individuals]
    fitnesses = [ind.gene.fitness for ind in individuals]
    complexities = [ind.gene.complexity() for ind in individuals]
    
    # 1. Hidden Dim vs Fitness
    scatter = axes[0, 0].scatter(hidden_dims, fitnesses, c=complexities, 
                                 s=100, cmap='viridis', alpha=0.6, edgecolors='black')
    axes[0, 0].set_xlabel('Hidden Dimension', fontsize=11)
    axes[0, 0].set_ylabel('Fitness', fontsize=11)
    axes[0, 0].set_title('Hidden Dimension vs Fitness', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Complexity')
    
    # 2. Encoder Layers vs Fitness
    axes[0, 1].scatter(enc_layers, fitnesses, c=hidden_dims, 
                      s=100, cmap='plasma', alpha=0.6, edgecolors='black')
    axes[0, 1].set_xlabel('Encoder Layers', fontsize=11)
    axes[0, 1].set_ylabel('Fitness', fontsize=11)
    axes[0, 1].set_title('Encoder Layers vs Fitness', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Complexity vs Fitness (Pareto Front)
    axes[1, 0].scatter(complexities, fitnesses, s=100, alpha=0.6, 
                      c='green', edgecolors='black')
    axes[1, 0].set_xlabel('Model Complexity (approx params)', fontsize=11)
    axes[1, 0].set_ylabel('Fitness', fontsize=11)
    axes[1, 0].set_title('Complexity vs Fitness (Pareto Front)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight Pareto-optimal (best fitness for given complexity)
    sorted_idx = np.argsort(complexities)
    sorted_complex = np.array(complexities)[sorted_idx]
    sorted_fitness = np.array(fitnesses)[sorted_idx]
    
    pareto_front = []
    max_fitness = -np.inf
    for c, f in zip(sorted_complex, sorted_fitness):
        if f > max_fitness:
            pareto_front.append((c, f))
            max_fitness = f
    
    if pareto_front:
        pareto_x, pareto_y = zip(*pareto_front)
        axes[1, 0].plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Front')
        axes[1, 0].legend()
    
    # 4. Learning Rate Distribution
    lr_bins = np.logspace(np.log10(min(learning_rates)), 
                          np.log10(max(learning_rates)), 10)
    axes[1, 1].hist(learning_rates, bins=lr_bins, color='orange', 
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Learning Rate', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Learning Rate Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Architecture comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_evolution_summary(neuro_evo):
    """
    Print detailed summary of evolution results.
    """
    print(f"\n{'='*80}")
    print(f"NEUROEVOLUTION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f" Evolution Statistics:")
    print(f"   Generations completed: {neuro_evo.generation}")
    print(f"   Population size: {neuro_evo.pop_size}")
    print(f"   Elite size: {neuro_evo.elite_size}")
    print(f"   Mutation rate: {neuro_evo.mutation_rate}")
    print(f"   Crossover rate: {neuro_evo.crossover_rate}")
    
    if neuro_evo.fitness_history:
        print(f"\n Fitness Progression:")
        first_gen = neuro_evo.fitness_history[0]
        last_gen = neuro_evo.fitness_history[-1]
        
        improvement = last_gen['best_fitness'] - first_gen['best_fitness']
        improvement_pct = (improvement / abs(first_gen['best_fitness'])) * 100 if first_gen['best_fitness'] != 0 else 0
        
        print(f"   Initial best fitness: {first_gen['best_fitness']:.4f}")
        print(f"   Final best fitness: {last_gen['best_fitness']:.4f}")
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"   Initial mean fitness: {first_gen['mean_fitness']:.4f} ± {first_gen['std_fitness']:.4f}")
        print(f"   Final mean fitness: {last_gen['mean_fitness']:.4f} ± {last_gen['std_fitness']:.4f}")
    
    if neuro_evo.best_individual:
        best = neuro_evo.best_individual
        print(f"\n Best Architecture Found:")
        print(f"   Fitness: {best.gene.fitness:.4f}")
        print(f"   Utilization: {best.avg_utilization:.3f}")
        print(f"   Bins used: {best.avg_bins_used:.2f}")
        print(f"   Items placed: {best.avg_items_placed:.1%}")
        print(f"\n   Architecture Parameters:")
        print(f"   ├─ Hidden dim: {best.gene.hidden_dim}")
        print(f"   ├─ Encoder layers: {best.gene.enc_layers}")
        print(f"   ├─ Head hidden: {best.gene.head_hidden}")
        print(f"   ├─ Learning rate: {best.gene.lr:.0e}")
        print(f"   ├─ Gamma: {best.gene.gamma}")
        print(f"   ├─ N-step: {best.gene.n_step}")
        print(f"   ├─ Batch size: {best.gene.batch_size}")
        print(f"   └─ Complexity: ~{best.gene.complexity():,} parameters")
    
    # Population statistics
    if neuro_evo.population:
        print(f"\n🧬 Population Statistics (Generation {neuro_evo.generation}):")
        
        fitnesses = [ind.gene.fitness for ind in neuro_evo.population]
        hidden_dims = [ind.gene.hidden_dim for ind in neuro_evo.population]
        layers = [ind.gene.enc_layers for ind in neuro_evo.population]
        
        print(f"   Fitness range: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]")
        print(f"   Hidden dim range: [{min(hidden_dims)}, {max(hidden_dims)}]")
        print(f"   Layers range: [{min(layers)}, {max(layers)}]")
        
        # Most common architectures
        from collections import Counter
        arch_strings = [f"h={ind.gene.hidden_dim},l={ind.gene.enc_layers}" 
                       for ind in neuro_evo.population]
        common = Counter(arch_strings).most_common(3)
        
        print(f"\n   Most common architectures:")
        for i, (arch, count) in enumerate(common, 1):
            print(f"   {i}. {arch} (×{count})")
    
    print(f"\n{'='*80}\n")


def compare_manual_vs_evolved(manual_config, evolved_config, 
                              manual_results, evolved_results):
    """
    Compare manually designed vs evolved architecture.
    """
    print(f"\n{'='*80}")
    print(f"MANUAL vs EVOLVED ARCHITECTURE COMPARISON")
    print(f"{'='*80}\n")
    
    print(f" Architecture Comparison:")
    print(f"\n   Manual Design:")
    print(f"   ├─ Hidden: {manual_config['hidden_dim']}")
    print(f"   ├─ Layers: {manual_config['enc_layers']}")
    print(f"   ├─ Head hidden: {manual_config['head_hidden']}")
    print(f"   ├─ LR: {manual_config['lr']:.0e}")
    print(f"   └─ Complexity: ~{manual_config['hidden_dim']**2 * manual_config['enc_layers']:,} params")
    
    print(f"\n   Evolved Design:")
    print(f"   ├─ Hidden: {evolved_config['hidden_dim']}")
    print(f"   ├─ Layers: {evolved_config['enc_layers']}")
    print(f"   ├─ Head hidden: {evolved_config['head_hidden']}")
    print(f"   ├─ LR: {evolved_config['lr']:.0e}")
    print(f"   └─ Complexity: ~{evolved_config['hidden_dim']**2 * evolved_config['enc_layers']:,} params")
    
    print(f"\n Performance Comparison:")
    
    metrics = ['utilization', 'bins_used', 'items_placed']
    for metric in metrics:
        manual_val = manual_results.get(metric, 0)
        evolved_val = evolved_results.get(metric, 0)
        
        if metric == 'bins_used':
            diff = manual_val - evolved_val  # Lower is better
            better = "Evolved" if diff > 0 else "Manual"
        else:
            diff = evolved_val - manual_val  # Higher is better
            better = "Evolved" if diff > 0 else "Manual"
        
        print(f"\n   {metric.replace('_', ' ').title()}:")
        print(f"   ├─ Manual: {manual_val:.3f}")
        print(f"   ├─ Evolved: {evolved_val:.3f}")
        print(f"   ├─ Difference: {diff:+.3f}")
        print(f"   └─ Winner: {better}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    print("Neuroevolution Visualization Tools")
    print("===================================\n")
    
    print("These tools are meant to be imported and used with:")
    print("  from neuroevolution_viz import plot_comprehensive_evolution")
    print("  plot_comprehensive_evolution(neuro_evo, 'output.png')")
    print("\nAvailable functions:")
    print("  - plot_comprehensive_evolution(neuro_evo)")
    print("  - plot_architecture_comparison(individuals)")
    print("  - print_evolution_summary(neuro_evo)")
    print("  - compare_manual_vs_evolved(configs, results)")
