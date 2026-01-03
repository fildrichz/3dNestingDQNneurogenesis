"""
Experiment Pipeline Visualization
Generates a comprehensive diagram of the 3D Bin Packing DQN with Neurogenesis pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

def create_pipeline_diagram():
    """Create comprehensive experiment pipeline visualization"""

    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Color scheme
    colors = {
        'entry': '#FF6B6B',      # Red - Entry point
        'data': '#4ECDC4',       # Teal - Data
        'ga': '#95E1D3',         # Light teal - GA
        'genome': '#FFE66D',     # Yellow - Genome
        'nn': '#FF8C42',         # Orange - Neural Network
        'env': '#A8E6CF',        # Green - Environment
        'training': '#C7CEEA',   # Purple - Training
        'eval': '#FFDAC1',       # Peach - Evaluation
        'output': '#B4F8C8'      # Light green - Output
    }

    # Title
    ax.text(10, 27, 'Experiment Pipeline: Multi-Problem Neural Architecture Evolution',
            fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))

    y_pos = 25.5

    # ====== ENTRY POINT ======
    ax.add_patch(FancyBboxPatch((7, y_pos-0.8), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['entry'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.5, 'experiment_multi_problem.py\n(Main Entry Point)',
            fontsize=11, fontweight='bold', ha='center', va='center')

    # Arrow down
    ax.add_patch(FancyArrowPatch((10, y_pos-0.8), (10, y_pos-1.5),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black'))

    y_pos -= 2.3

    # ====== PHASE 0: DATA LOADING ======
    ax.text(10, y_pos, 'PHASE 0: Data Loading', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=2))

    y_pos -= 0.8

    # Dataset box
    ax.add_patch(FancyBboxPatch((3, y_pos-1.2), 5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['data'],
                                edgecolor='black', linewidth=2))
    ax.text(5.5, y_pos-0.7, 'Training Problems\n(IDs: 1,2,3,5,7,...)',
            fontsize=10, ha='center', va='center')

    ax.add_patch(FancyBboxPatch((12, y_pos-1.2), 5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['data'],
                                edgecolor='black', linewidth=2))
    ax.text(14.5, y_pos-0.7, 'Test Problems\n(IDs: 12,13,...)',
            fontsize=10, ha='center', va='center')

    # Dataset loader
    ax.add_patch(FancyBboxPatch((8, y_pos-2), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['data'],
                                edgecolor='black', linewidth=1.5))
    ax.text(10, y_pos-1.7, 'dataset_loader.py\nBinPackingProblem',
            fontsize=9, ha='center', va='center')

    # Arrows
    ax.add_patch(FancyArrowPatch((5.5, y_pos-1.2), (9, y_pos-1.6),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black'))
    ax.add_patch(FancyArrowPatch((14.5, y_pos-1.2), (11, y_pos-1.6),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black'))

    y_pos -= 2.8

    # Arrow to Phase 1
    ax.add_patch(FancyArrowPatch((10, y_pos+0.6), (10, y_pos),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black'))

    # ====== PHASE 1: GENETIC ALGORITHM EVOLUTION ======
    ax.text(10, y_pos, 'PHASE 1: Architecture Evolution (GA)', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['ga'], edgecolor='black', linewidth=2))

    y_pos -= 0.8

    # GA Controller
    ax.add_patch(FancyBboxPatch((7, y_pos-0.8), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['ga'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.5, 'ga_evolution.py\nevolve_multi_problem_architecture()',
            fontsize=10, fontweight='bold', ha='center', va='center')

    y_pos -= 1.5

    # Population initialization
    ax.add_patch(FancyBboxPatch((1, y_pos-0.8), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['genome'],
                                edgecolor='black', linewidth=1.5))
    ax.text(3, y_pos-0.5, 'Initialize Population\n(20-50 genomes)',
            fontsize=9, ha='center', va='center')

    # Genome structure
    ax.add_patch(FancyBboxPatch((6, y_pos-1.8), 8, 1.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['genome'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.3, 'NetworkGenome (genome.py)', fontsize=10, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.7, 'Architecture Genes:', fontsize=9, ha='center', style='italic')
    ax.text(10, y_pos-1, '• hidden_dim, enc_layers, head_hidden', fontsize=8, ha='center')
    ax.text(10, y_pos-1.25, '• attention_type, attention_heads, num_inducing_points', fontsize=8, ha='center')
    ax.text(10, y_pos-1.5, '• patch_size, cnn_channels, dropout, activation', fontsize=8, ha='center')

    # Arrow to genome
    ax.add_patch(FancyArrowPatch((5, y_pos-0.5), (6, y_pos-0.5),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black'))

    y_pos -= 2.5

    # GA Loop box
    ax.add_patch(Rectangle((0.5, y_pos-5.5), 19, 5,
                           facecolor='white',
                           edgecolor='blue', linewidth=3, linestyle='--'))
    ax.text(10, y_pos-0.2, 'For each Generation (50 generations):',
            fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', edgecolor='blue'))

    y_pos -= 0.8

    # Evaluation step
    ax.add_patch(FancyBboxPatch((2, y_pos-1), 7, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['training'],
                                edgecolor='black', linewidth=2))
    ax.text(5.5, y_pos-0.3, 'For each Genome:', fontsize=9, fontweight='bold', ha='center')
    ax.text(5.5, y_pos-0.7, 'evaluate_genome_multi_problem()', fontsize=9, ha='center')

    # Training details
    ax.add_patch(FancyBboxPatch((10, y_pos-1.4), 8.5, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['training'],
                                edgecolor='black', linewidth=1.5))
    ax.text(14.25, y_pos-0.5, 'Round-Robin Training:', fontsize=9, fontweight='bold', ha='center')
    ax.text(14.25, y_pos-0.85, '• Train on Problem 1, then Problem 2, ...', fontsize=8, ha='center')
    ax.text(14.25, y_pos-1.15, '• 100 episodes total (distributed across problems)', fontsize=8, ha='center')

    # Arrow
    ax.add_patch(FancyArrowPatch((9, y_pos-0.6), (10, y_pos-0.6),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black'))

    y_pos -= 1.8

    # Fitness calculation
    ax.add_patch(FancyBboxPatch((10, y_pos-0.8), 8.5, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['eval'],
                                edgecolor='black', linewidth=1.5))
    ax.text(14.25, y_pos-0.5, 'Fitness = avg(utilization) - complexity_penalty',
            fontsize=9, ha='center', va='center')

    y_pos -= 1.2

    # Selection
    ax.add_patch(FancyBboxPatch((2, y_pos-0.6), 4, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['ga'],
                                edgecolor='black', linewidth=1.5))
    ax.text(4, y_pos-0.35, 'Tournament Selection', fontsize=9, ha='center', va='center')

    # Crossover
    ax.add_patch(FancyBboxPatch((7, y_pos-0.6), 3, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['ga'],
                                edgecolor='black', linewidth=1.5))
    ax.text(8.5, y_pos-0.35, 'Crossover', fontsize=9, ha='center', va='center')

    # Mutation
    ax.add_patch(FancyBboxPatch((11, y_pos-0.6), 3, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['ga'],
                                edgecolor='black', linewidth=1.5))
    ax.text(12.5, y_pos-0.35, 'Mutation', fontsize=9, ha='center', va='center')

    # Elitism
    ax.add_patch(FancyBboxPatch((15, y_pos-0.6), 3.5, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['ga'],
                                edgecolor='black', linewidth=1.5))
    ax.text(16.75, y_pos-0.35, 'Elitism (top 3)', fontsize=9, ha='center', va='center')

    # Arrow showing loop
    ax.annotate('', xy=(1.5, y_pos-5.3), xytext=(1.5, y_pos),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(0.5, y_pos-2.5, 'Loop\nback', fontsize=9, ha='center', color='blue', fontweight='bold')

    y_pos -= 1.5

    # Best genome output
    ax.add_patch(FancyBboxPatch((7, y_pos-0.8), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.5, 'Best Evolved Genome\n(highest fitness)',
            fontsize=10, fontweight='bold', ha='center', va='center')

    y_pos -= 1.5

    # Arrow to Phase 2
    ax.add_patch(FancyArrowPatch((10, y_pos+0.5), (10, y_pos),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black'))

    # ====== PHASE 2: FINAL TRAINING ======
    ax.text(10, y_pos, 'PHASE 2: Final Model Training', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['training'], edgecolor='black', linewidth=2))

    y_pos -= 1

    # Build DQN agent
    ax.add_patch(FancyBboxPatch((1, y_pos-1), 8, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['nn'],
                                edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.3, 'Build DQN Agent from Best Genome', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, y_pos-0.7, 'dqn_enhanced.py → DQNAgentEnhanced', fontsize=9, ha='center')

    # Neural network architecture
    ax.add_patch(FancyBboxPatch((10, y_pos-2.2), 8.5, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['nn'],
                                edgecolor='black', linewidth=2))
    ax.text(14.25, y_pos-0.4, 'Q-Network Architecture:', fontsize=10, fontweight='bold', ha='center')
    ax.text(14.25, y_pos-0.75, '1. State Encoder MLP', fontsize=9, ha='center')
    ax.text(14.25, y_pos-1.05, '2. HeightmapCNN (7×7 patches → 64-dim)', fontsize=9, ha='center')
    ax.text(14.25, y_pos-1.35, '3. Action Encoder MLP', fontsize=9, ha='center')
    ax.text(14.25, y_pos-1.65, '4. Transformer Attention (action relationships)', fontsize=9, ha='center')
    ax.text(14.25, y_pos-1.95, '5. Q-Value Head', fontsize=9, ha='center')

    y_pos -= 2.5

    # Training loop
    ax.add_patch(FancyBboxPatch((3, y_pos-0.8), 14, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['training'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.5, 'Train for 2000 Episodes (Round-Robin across Training Problems)',
            fontsize=10, fontweight='bold', ha='center', va='center')

    y_pos -= 1.3

    # Environment
    ax.add_patch(FancyBboxPatch((1, y_pos-1.8), 8, 1.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['env'],
                                edgecolor='black', linewidth=2))
    ax.text(5, y_pos-0.3, 'MultiBinPackingEnv', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, y_pos-0.7, 'packing_with_dqncore2_enhanced.py', fontsize=9, ha='center', style='italic')
    ax.text(5, y_pos-1.05, '• EMS-based action generation', fontsize=8, ha='center')
    ax.text(5, y_pos-1.3, '• Constraint validation (weight, affinity, etc.)', fontsize=8, ha='center')
    ax.text(5, y_pos-1.55, '• Heightmap extraction', fontsize=8, ha='center')

    # Replay buffer
    ax.add_patch(FancyBboxPatch((10, y_pos-1.2), 8.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['training'],
                                edgecolor='black', linewidth=1.5))
    ax.text(14.25, y_pos-0.5, 'Experience Replay Buffer', fontsize=9, fontweight='bold', ha='center')
    ax.text(14.25, y_pos-0.85, '• Stores (s, a, r, s\', done)', fontsize=8, ha='center')
    ax.text(14.25, y_pos-1.1, '• + action features + heightmap patches', fontsize=8, ha='center')

    y_pos -= 2.3

    # Trained model output
    ax.add_patch(FancyBboxPatch((7, y_pos-0.8), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.5, 'Trained Model (.pth)\n+ Genome Config (.json)',
            fontsize=10, fontweight='bold', ha='center', va='center')

    y_pos -= 1.5

    # Arrow to Phase 3
    ax.add_patch(FancyArrowPatch((10, y_pos+0.5), (10, y_pos),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='black'))

    # ====== PHASE 3: EVALUATION ======
    ax.text(10, y_pos, 'PHASE 3: Evaluation on Test Problems', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['eval'], edgecolor='black', linewidth=2))

    y_pos -= 0.8

    # Evaluation box
    ax.add_patch(FancyBboxPatch((4, y_pos-1), 12, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['eval'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.3, 'Evaluate Trained Agent on Test Problems', fontsize=10, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.7, '10 episodes per problem, ε=0 (greedy)', fontsize=9, ha='center')

    y_pos -= 1.5

    # Metrics
    ax.add_patch(FancyBboxPatch((2, y_pos-1.2), 7, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['eval'],
                                edgecolor='black', linewidth=1.5))
    ax.text(5.5, y_pos-0.4, 'Performance Metrics:', fontsize=9, fontweight='bold', ha='center')
    ax.text(5.5, y_pos-0.7, '• Bin utilization', fontsize=8, ha='center')
    ax.text(5.5, y_pos-0.95, '• Bins used', fontsize=8, ha='center')

    # Visualizations
    ax.add_patch(FancyBboxPatch((10.5, y_pos-1.2), 7.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=1.5))
    ax.text(14.25, y_pos-0.4, '3D Visualizations:', fontsize=9, fontweight='bold', ha='center')
    ax.text(14.25, y_pos-0.7, '• Bin packing plots (filled)', fontsize=8, ha='center')
    ax.text(14.25, y_pos-0.95, '• EMS visualization', fontsize=8, ha='center')

    y_pos -= 1.8

    # Final outputs
    ax.add_patch(FancyBboxPatch((5, y_pos-1.2), 10, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2))
    ax.text(10, y_pos-0.3, 'Final Outputs:', fontsize=10, fontweight='bold', ha='center')
    ax.text(10, y_pos-0.6, '• results.json (test metrics)', fontsize=9, ha='center')
    ax.text(10, y_pos-0.85, '• evolution_history.json (GA convergence)', fontsize=9, ha='center')
    ax.text(10, y_pos-1.05, '• visualizations/ (bin packing 3D plots)', fontsize=9, ha='center')

    # Add legend for components
    y_legend = 0.5
    legend_items = [
        ('Entry Point', colors['entry']),
        ('Data', colors['data']),
        ('GA Evolution', colors['ga']),
        ('Genome', colors['genome']),
        ('Neural Network', colors['nn']),
        ('Environment', colors['env']),
        ('Training', colors['training']),
        ('Evaluation', colors['eval']),
        ('Output', colors['output'])
    ]

    x_legend = 0.5
    for label, color in legend_items:
        ax.add_patch(Rectangle((x_legend, y_legend), 0.4, 0.3,
                              facecolor=color, edgecolor='black', linewidth=1))
        ax.text(x_legend + 0.5, y_legend + 0.15, label, fontsize=8, va='center')
        x_legend += 2.2

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Generating experiment pipeline diagram...")
    fig = create_pipeline_diagram()

    # Save as PNG
    output_path = '/home/user/3dNestingDQNneurogenesis/docs/experiment_pipeline_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Diagram saved to: {output_path}")

    # Also save as PDF for better quality
    output_path_pdf = '/home/user/3dNestingDQNneurogenesis/docs/experiment_pipeline_diagram.pdf'
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {output_path_pdf}")

    plt.close()
    print("Done!")
