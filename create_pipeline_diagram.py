#!/usr/bin/env python3
"""
Generate a pipeline diagram for the thesis presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'input': '#E3F2FD',      # Light blue
    'ga': '#FFF3E0',         # Light orange
    'dqn': '#E8F5E9',        # Light green
    'output': '#F3E5F5',     # Light purple
    'arrow': '#455A64',      # Dark gray
    'border_ga': '#E65100',  # Orange
    'border_dqn': '#2E7D32', # Green
    'border_input': '#1565C0', # Blue
    'border_output': '#7B1FA2', # Purple
}

def draw_box(ax, x, y, w, h, text, color, border_color, fontsize=10, bold=False):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#455A64'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(7, 9.5, '3D Bin Packing: DQN + Neuroevolution Pipeline',
        ha='center', va='center', fontsize=16, fontweight='bold')

# ========== INPUT SECTION ==========
draw_box(ax, 0.5, 7, 2.5, 1.2, 'Problem Instance\n(3dBPP_*.txt)',
         colors['input'], colors['border_input'], fontsize=9, bold=True)

draw_box(ax, 0.5, 5.3, 2.5, 1.2, 'Items + Constraints\n(weight, incomp.)',
         colors['input'], colors['border_input'], fontsize=9)

# Arrow from problem to items
draw_arrow(ax, (1.75, 7), (1.75, 6.6))

# ========== GA OUTER LOOP ==========
# Big box for GA
ga_box = FancyBboxPatch((3.5, 2.5), 7, 5.5,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor='#FFF8E1', edgecolor=colors['border_ga'],
                         linewidth=3, linestyle='--')
ax.add_patch(ga_box)
ax.text(7, 7.7, 'GENETIC ALGORITHM (Outer Loop)',
        ha='center', va='center', fontsize=12, fontweight='bold', color=colors['border_ga'])

# Population
draw_box(ax, 3.8, 6.3, 2.2, 1, 'Population\n(30 genomes)',
         colors['ga'], colors['border_ga'], fontsize=9, bold=True)

# Genome representation
draw_box(ax, 6.3, 6.3, 2.4, 1, '17 Genes\n(arch + hyperparams)',
         colors['ga'], colors['border_ga'], fontsize=9)

# Selection/Crossover/Mutation
draw_box(ax, 8.9, 6.3, 1.4, 1, 'Selection\nCrossover\nMutation',
         colors['ga'], colors['border_ga'], fontsize=8)

# Arrows in GA loop
draw_arrow(ax, (6.0, 6.8), (6.3, 6.8), colors['border_ga'])
draw_arrow(ax, (8.7, 6.8), (8.9, 6.8), colors['border_ga'])

# ========== DQN INNER LOOP ==========
# Box for DQN
dqn_box = FancyBboxPatch((4, 2.8), 6.2, 3,
                          boxstyle="round,pad=0.02,rounding_size=0.2",
                          facecolor='#E8F5E9', edgecolor=colors['border_dqn'],
                          linewidth=2)
ax.add_patch(dqn_box)
ax.text(7.1, 5.5, 'DQN TRAINING (Inner Loop)',
        ha='center', va='center', fontsize=11, fontweight='bold', color=colors['border_dqn'])

# DQN components
draw_box(ax, 4.3, 4.2, 1.8, 0.9, 'State\nEncoder',
         '#C8E6C9', colors['border_dqn'], fontsize=9)

draw_box(ax, 6.2, 4.2, 1.8, 0.9, 'Heightmap\nCNN',
         '#C8E6C9', colors['border_dqn'], fontsize=9)

draw_box(ax, 8.1, 4.2, 1.8, 0.9, 'Attention\n(evolved)',
         '#C8E6C9', colors['border_dqn'], fontsize=9)

# Q-value output
draw_box(ax, 6.2, 3.1, 1.8, 0.8, 'Q(s,a)',
         '#A5D6A7', colors['border_dqn'], fontsize=10, bold=True)

# Arrows to Q
draw_arrow(ax, (5.2, 4.2), (6.5, 3.9), colors['border_dqn'])
draw_arrow(ax, (7.1, 4.2), (7.1, 3.9), colors['border_dqn'])
draw_arrow(ax, (9, 4.2), (7.7, 3.9), colors['border_dqn'])

# ========== FITNESS EVALUATION ==========
draw_box(ax, 4.3, 0.8, 2.5, 1.2, 'Fitness Evaluation\n0.7×util + 0.2×bins\n+ 0.1×complexity',
         colors['ga'], colors['border_ga'], fontsize=8, bold=True)

# Arrow from Q to fitness
draw_arrow(ax, (7.1, 3.1), (5.5, 2.1))

# Feedback arrow (curved) from fitness back to population
ax.annotate('', xy=(4.9, 6.3), xytext=(4.3, 2.0),
            arrowprops=dict(arrowstyle='->', color=colors['border_ga'],
                          lw=2, connectionstyle='arc3,rad=-0.4'))

# ========== OUTPUT SECTION ==========
draw_box(ax, 11, 5.5, 2.5, 1.2, 'Best Architecture\n(evolved genome)',
         colors['output'], colors['border_output'], fontsize=9, bold=True)

draw_box(ax, 11, 3.8, 2.5, 1.2, 'Trained Model\n(.pth weights)',
         colors['output'], colors['border_output'], fontsize=9)

draw_box(ax, 11, 2.1, 2.5, 1.2, 'Packing Solution\n(visualization)',
         colors['output'], colors['border_output'], fontsize=9)

# Arrows from GA to output
draw_arrow(ax, (10.5, 6.8), (11, 6.1))
draw_arrow(ax, (10.2, 3.5), (11, 4.4))

# Arrow from input to GA
draw_arrow(ax, (3, 6.5), (3.8, 6.8))

# ========== LEGEND ==========
legend_y = 0.3
ax.add_patch(FancyBboxPatch((0.5, legend_y), 0.4, 0.3,
             facecolor=colors['input'], edgecolor=colors['border_input'], linewidth=1.5))
ax.text(1.1, legend_y + 0.15, 'Input', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((2, legend_y), 0.4, 0.3,
             facecolor=colors['ga'], edgecolor=colors['border_ga'], linewidth=1.5))
ax.text(2.6, legend_y + 0.15, 'Genetic Algorithm', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((4.5, legend_y), 0.4, 0.3,
             facecolor='#C8E6C9', edgecolor=colors['border_dqn'], linewidth=1.5))
ax.text(5.1, legend_y + 0.15, 'DQN Components', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((7.2, legend_y), 0.4, 0.3,
             facecolor=colors['output'], edgecolor=colors['border_output'], linewidth=1.5))
ax.text(7.8, legend_y + 0.15, 'Output', fontsize=8, va='center')

# Generation counter annotation
ax.text(3.7, 2.3, '50 generations', fontsize=8, style='italic', color=colors['border_ga'])
ax.text(4.2, 5.2, '100 episodes/eval', fontsize=8, style='italic', color=colors['border_dqn'])

plt.tight_layout()
plt.savefig('/home/user/3dNestingDQNneurogenesis/pipeline_diagram.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/home/user/3dNestingDQNneurogenesis/pipeline_diagram.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Pipeline diagram saved to pipeline_diagram.png and pipeline_diagram.pdf")
