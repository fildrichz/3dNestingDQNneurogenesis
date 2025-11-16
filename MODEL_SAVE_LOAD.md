# Model Save/Load Documentation

This guide explains how to save, load, and continue training models across different datasets.

## Table of Contents
- [Quick Start](#quick-start)
- [Saving Models](#saving-models)
- [Loading Models](#loading-models)
- [Transfer Learning](#transfer-learning)
- [Checkpoint Contents](#checkpoint-contents)
- [Examples](#examples)

---

## Quick Start

### Save a Model
```python
# During training (automatic save)
train_with_evolved_genome(
    problem_path=full_datapath('3dBPP_4.txt'),
    genome=best_genome,
    episodes=200,
    save_path='output_data/my_model.pth'
)
```

### Load for Evaluation
```python
from example_ga_training import load_trained_model

agent, genome, checkpoint = load_trained_model(
    model_path='output_data/my_model.pth',
    problem_path=full_datapath('3dBPP_4.txt'),
    for_training=False  # Evaluation mode
)
```

### Continue Training on New Dataset
```python
from example_ga_training import continue_training_on_new_dataset

agent, genome = continue_training_on_new_dataset(
    model_path='output_data/my_model.pth',
    new_problem_path=full_datapath('3dBPP_5.txt'),
    episodes=200,
    save_path='output_data/my_model_continued.pth'
)
```

---

## Saving Models

### Method 1: Using `example_ga_training.py`

**Automatic save during training:**
```python
agent = train_with_evolved_genome(
    problem_path=full_datapath('3dBPP_4.txt'),
    genome=best_genome,
    episodes=200,
    save_path='output_data/evolved_model.pth'  # Will save automatically
)
```

**What gets saved:**
- Model weights (Q-network)
- Target network weights
- Optimizer state (Adam parameters)
- Training progress (env_steps, training_steps)
- Exploration rate (epsilon)
- Network architecture (genome)
- Configuration parameters
- Performance metrics (best_bins, best_items, best_util)
- Problem file path

### Method 2: Using `packing_with_dqncore2_enhanced.py`

**Automatic save during training:**
```python
from packing_with_dqncore2_enhanced import train_multibin_pack_dqn

agent, env, best_solution = train_multibin_pack_dqn(
    problem_path=full_datapath('3dBPP_4.txt'),
    episodes=400,
    save_path='my_model.pth'  # Automatically prefixed with 'output_data/'
)
```

**Manual save:**
```python
from packing_with_dqncore2_enhanced import save_model

save_model(
    agent=agent,
    save_path='output_data/custom_model.pth',
    problem_info={
        'bin_dimensions': (10, 10, 10),
        'num_items': 50
    },
    training_stats={
        'best_bins': 5,
        'best_util': 0.85
    }
)
```

---

## Loading Models

### For Evaluation Only

Load model weights without training state (sets model to eval mode):

```python
from example_ga_training import load_trained_model

agent, genome, checkpoint = load_trained_model(
    model_path='output_data/evolved_model.pth',
    problem_path=full_datapath('3dBPP_4.txt'),
    device='cuda',  # Optional: 'cuda' or 'cpu', auto-detects if None
    for_training=False  # Evaluation mode
)

# Agent is ready for evaluation
agent.q.eval()  # Already in eval mode
```

### For Continuing Training

Load model with full training state (optimizer, epsilon, etc.):

```python
from example_ga_training import load_trained_model

agent, genome, checkpoint = load_trained_model(
    model_path='output_data/evolved_model.pth',
    problem_path=full_datapath('3dBPP_4.txt'),
    device='cuda',
    for_training=True  # Restores optimizer, epsilon, training steps
)

# Agent ready to continue training
print(f"Current epsilon: {agent.epsilon}")
print(f"Training steps: {agent.training_steps}")
```

### Using `packing_with_dqncore2_enhanced.py`

```python
from packing_with_dqncore2_enhanced import load_model

agent, checkpoint, env = load_model(
    load_path='output_data/my_model.pth',
    problem_path=full_datapath('3dBPP_4.txt'),  # Optional
    device='cuda'
)

# If problem_path provided, env is created automatically
# Otherwise, env will be None
```

---

## Transfer Learning

Transfer learning allows you to train on one dataset, then continue training on different datasets. This is useful for:
- **Curriculum learning**: Train on easy problems first, then harder ones
- **Multi-dataset training**: Learn general packing strategies across problems
- **Fine-tuning**: Adapt a pre-trained model to specific problem characteristics

### Basic Transfer Learning

```python
from example_ga_training import continue_training_on_new_dataset

# Step 1: Load model trained on dataset A
# Step 2: Continue training on dataset B
agent, genome = continue_training_on_new_dataset(
    model_path='output_data/trained_on_A.pth',
    new_problem_path=full_datapath('3dBPP_B.txt'),
    episodes=200,
    save_path='output_data/trained_on_B.pth',
    reset_epsilon=False,  # Keep current exploration rate
    device='cuda'
)
```

### Parameters

- **model_path**: Path to previously saved checkpoint
- **new_problem_path**: New problem file to train on
- **episodes**: Number of additional training episodes
- **save_path**: Where to save updated model (None = don't save)
- **reset_epsilon**:
  - `False` (default): Keep current epsilon (good for similar problems)
  - `True`: Reset to initial epsilon (good for very different problems)
- **device**: 'cuda' or 'cpu'

### Sequential Multi-Dataset Training

```python
from example_ga_training import continue_training_on_new_dataset, train_with_evolved_genome

# Train on dataset 1
train_with_evolved_genome(
    problem_path=full_datapath('3dBPP_4.txt'),
    genome=best_genome,
    episodes=200,
    save_path='output_data/step1.pth'
)

# Continue on dataset 2
continue_training_on_new_dataset(
    model_path='output_data/step1.pth',
    new_problem_path=full_datapath('3dBPP_5.txt'),
    episodes=200,
    save_path='output_data/step2.pth'
)

# Continue on dataset 3
continue_training_on_new_dataset(
    model_path='output_data/step2.pth',
    new_problem_path=full_datapath('3dBPP_6.txt'),
    episodes=200,
    save_path='output_data/step3.pth'
)
```

### Curriculum Learning (Easy → Hard)

```python
# Start with easy problems
train_with_evolved_genome(
    problem_path=full_datapath('3dBPP_easy.txt'),
    genome=best_genome,
    episodes=100,
    save_path='output_data/easy_trained.pth'
)

# Progress to medium difficulty
continue_training_on_new_dataset(
    model_path='output_data/easy_trained.pth',
    new_problem_path=full_datapath('3dBPP_medium.txt'),
    episodes=200,
    reset_epsilon=True,  # More exploration for harder problem
    save_path='output_data/medium_trained.pth'
)

# Finally, hardest problems
continue_training_on_new_dataset(
    model_path='output_data/medium_trained.pth',
    new_problem_path=full_datapath('3dBPP_hard.txt'),
    episodes=300,
    reset_epsilon=True,
    save_path='output_data/hard_trained.pth'
)
```

---

## Checkpoint Contents

### example_ga_training.py Checkpoints

```python
checkpoint = {
    # Model state
    'model_state_dict': agent.q.state_dict(),           # Q-network weights
    'target_state_dict': agent.q_target.state_dict(),   # Target network weights
    'optimizer_state_dict': agent.opt.state_dict(),     # Optimizer (Adam) state

    # Training progress
    'env_steps': agent.env_steps,                       # Total environment steps
    'training_steps': agent.training_steps,             # Total training updates
    'epsilon': agent.epsilon,                           # Current exploration rate

    # Architecture
    'genome': genome.to_dict(),                         # Network architecture genes
    'config': cfg.__dict__,                             # Full DQN configuration

    # Performance metrics
    'best_bins': best_bins,                             # Best bins used
    'best_items': best_items,                           # Best items packed
    'best_util': best_util,                             # Best utilization

    # Metadata
    'problem_file': problem_path                        # Which dataset trained on
}
```

### packing_with_dqncore2_enhanced.py Checkpoints

```python
checkpoint = {
    # Model state
    'model_state_dict': agent.q.state_dict(),
    'target_state_dict': agent.q_target.state_dict(),
    'optimizer_state_dict': agent.opt.state_dict(),

    # Training progress
    'env_steps': agent.env_steps,
    'training_steps': agent.training_steps,
    'epsilon': agent.epsilon,

    # Configuration
    'config': agent.cfg.__dict__,

    # Optional metadata
    'problem_info': {                                   # Problem description
        'bin_dimensions': (W, D, H),
        'num_items': len(items),
        'max_bins': problem.max_bins,
        'max_weight': problem.max_weight
    },
    'training_stats': {                                 # Training statistics
        'best_bins': best_bins,
        'best_items': best_items,
        'best_util': best_util,
        'final_ma50_bins': np.mean(bins_hist),
        'final_ma50_items': np.mean(items_hist),
        'final_ma50_util': np.mean(util_hist)
    }
}
```

---

## Examples

### Example 1: Train, Save, Load, Evaluate

```python
from example_ga_training import train_with_evolved_genome, load_trained_model
from packing_with_dqncore2_enhanced import evaluate_agent_on_problem

# 1. Train and save
agent = train_with_evolved_genome(
    problem_path=full_datapath('3dBPP_4.txt'),
    genome=best_genome,
    episodes=200,
    save_path='output_data/my_model.pth'
)

# 2. Later... load and evaluate
agent, genome, checkpoint = load_trained_model(
    model_path='output_data/my_model.pth',
    problem_path=full_datapath('3dBPP_test.txt'),
    for_training=False
)

# 3. Evaluate on test set
metrics = evaluate_agent_on_problem(
    agent=agent,
    env=env,
    items=test_items,
    episodes=50,
    patch_size=genome.genes['patch_size']
)

print(f"Test utilization: {metrics['avg_utilization']:.3f}")
print(f"Test bins used: {metrics['avg_bins_used']:.1f}")
```

### Example 2: Cross-Dataset Training

```python
from example_ga_training import continue_training_on_new_dataset

datasets = [
    '3dBPP_4.txt',
    '3dBPP_5.txt',
    '3dBPP_6.txt',
    '3dBPP_7.txt'
]

current_model = 'output_data/initial_model.pth'

for i, dataset in enumerate(datasets[1:], 1):
    print(f"\n=== Training on dataset {i+1}/{len(datasets)} ===")

    agent, genome = continue_training_on_new_dataset(
        model_path=current_model,
        new_problem_path=full_datapath(dataset),
        episodes=150,
        save_path=f'output_data/model_step_{i+1}.pth',
        reset_epsilon=(i % 2 == 0)  # Reset epsilon every other dataset
    )

    current_model = f'output_data/model_step_{i+1}.pth'

print(f"\nFinal model: {current_model}")
```

### Example 3: Fine-tuning for Specific Problem

```python
from example_ga_training import load_trained_model, continue_training_on_new_dataset

# Load general pre-trained model
agent, genome, checkpoint = load_trained_model(
    model_path='output_data/general_model.pth',
    problem_path=full_datapath('specific_problem.txt'),
    for_training=True
)

print(f"Pre-trained on: {checkpoint.get('problem_file', 'unknown')}")
print(f"Pre-trained performance: {checkpoint['best_util']:.3f} util")

# Fine-tune on specific problem
agent, genome = continue_training_on_new_dataset(
    model_path='output_data/general_model.pth',
    new_problem_path=full_datapath('specific_problem.txt'),
    episodes=100,
    reset_epsilon=False,  # Keep learned exploration strategy
    save_path='output_data/finetuned_model.pth'
)
```

### Example 4: Model Versioning

```python
from pathlib import Path
import datetime

def save_versioned_model(agent, genome, base_name='model'):
    """Save model with timestamp version."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'output_data/{base_name}_{timestamp}.pth'

    # Save with torch.save
    torch.save({
        'model_state_dict': agent.q.state_dict(),
        'target_state_dict': agent.q_target.state_dict(),
        'optimizer_state_dict': agent.opt.state_dict(),
        'env_steps': agent.env_steps,
        'training_steps': agent.training_steps,
        'epsilon': agent.epsilon,
        'genome': genome.to_dict(),
        'config': agent.cfg.__dict__,
        'timestamp': timestamp
    }, save_path)

    print(f"Saved version: {save_path}")
    return save_path

# Usage
save_path = save_versioned_model(agent, genome, base_name='evolved')
# Output: output_data/evolved_20250116_143022.pth
```

---

## Best Practices

### 1. Always Save After Long Training
```python
# Good: Save after expensive training
train_with_evolved_genome(..., save_path='output_data/model.pth')

# Bad: Don't save, lose all training
train_with_evolved_genome(..., save_path=None)
```

### 2. Use Descriptive Names
```python
# Good
save_path='output_data/genome_standard_attention_3dBPP_4_200eps.pth'

# Less good
save_path='output_data/model1.pth'
```

### 3. Load with `for_training=True` When Continuing
```python
# Continuing training - need full state
agent, genome, checkpoint = load_trained_model(
    ..., for_training=True
)

# Just evaluating - don't need optimizer state
agent, genome, checkpoint = load_trained_model(
    ..., for_training=False
)
```

### 4. Track Model Provenance
```python
# Checkpoint automatically tracks which problem was trained on
print(f"Trained on: {checkpoint.get('problem_file', 'unknown')}")
print(f"Training steps: {checkpoint['training_steps']}")
print(f"Best performance: {checkpoint['best_util']:.3f}")
```

### 5. Reset Epsilon for Very Different Problems
```python
# Similar problem - keep epsilon
continue_training_on_new_dataset(..., reset_epsilon=False)

# Very different problem - reset epsilon
continue_training_on_new_dataset(..., reset_epsilon=True)
```

---

## Troubleshooting

### Issue: "KeyError: 'target_state_dict'"

**Cause**: Loading old checkpoint format without target network state.

**Solution**: The checkpoint was saved with old code. Use `for_training=False` or retrain.

```python
# Will work even with old checkpoints
agent, genome, checkpoint = load_trained_model(
    ..., for_training=False
)
```

### Issue: "RuntimeError: CUDA out of memory"

**Cause**: Model too large for GPU.

**Solution**: Load on CPU or use smaller batch size.

```python
agent, genome, checkpoint = load_trained_model(
    ..., device='cpu'
)
```

### Issue: Loaded model performs poorly

**Possible causes**:
1. Model loaded in eval mode when you wanted to continue training
2. Epsilon too low (not exploring enough)
3. Wrong problem dimensions

**Solutions**:
```python
# 1. Use for_training=True
agent, genome, checkpoint = load_trained_model(..., for_training=True)

# 2. Reset epsilon
agent._eps = agent.cfg.eps_start

# 3. Verify problem matches
print(f"Trained on: {checkpoint.get('problem_file')}")
```

---

## API Reference

### `train_with_evolved_genome(problem_path, genome, episodes, save_path)`
Train model with evolved architecture and save.

### `load_trained_model(model_path, problem_path, device, for_training)`
Load saved model checkpoint.

### `continue_training_on_new_dataset(model_path, new_problem_path, episodes, save_path, reset_epsilon, device)`
Transfer learning: continue training on new dataset.

### `save_model(agent, save_path, problem_info, training_stats)`
Manually save model (packing_with_dqncore2_enhanced.py).

### `load_model(load_path, problem_path, device)`
Load model (packing_with_dqncore2_enhanced.py).

---

## Related Files

- `example_ga_training.py`: GA-based architecture evolution with save/load
- `packing_with_dqncore2_enhanced.py`: DQN training with save/load
- `dqn_core/dqn_enhanced.py`: DQNAgentEnhanced class with save()/load() methods
- `genome.py`: NetworkGenome architecture definition

---

*Last updated: 2025-01-16*
