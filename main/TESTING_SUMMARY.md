# Model Testing Scripts - Quick Reference

## Created Files

1. **`test_model.py`** - Single problem testing script
2. **`test_model_batch.py`** - Batch testing script for multiple problems
3. **`TEST_README.md`** - Comprehensive documentation
4. **`example_test_usage.sh`** - Example usage script
5. **`TESTING_SUMMARY.md`** - This quick reference

## Quick Start

**Prerequisites:** Place your trained model and genome in `trainedModels/` folder:
- `trainedModels/trained_model.pth`
- `trainedModels/evolved_genome.json`

### Test a single problem (using defaults):
```bash
python test_model.py
```

### Test a specific problem by ID:
```bash
python test_model.py --problem-id 4  # Tests 3dBPP_4.txt
```

### Test with visualization:
```bash
python test_model.py --visualize --num-episodes 10
```

### Test multiple problems (using defaults: problems 1-5):
```bash
python test_model_batch.py
```

### Test custom problem set:
```bash
python test_model_batch.py --problem-ids 1 4 12  # Test problems
```

### Run examples:
```bash
./example_test_usage.sh
```

## Key Features

- ✅ Load specific trained models
- ✅ Test on specific problems
- ✅ Multiple evaluation episodes for statistical significance
- ✅ 3D visualization generation
- ✅ Detailed metrics (utilization, bins used, success rate)
- ✅ Batch testing on multiple problems
- ✅ JSON and CSV export
- ✅ Same nesting algorithm as training (faithful evaluation)
- ✅ GPU and CPU support

## Output Metrics

- **Utilization**: Volume efficiency (packed_volume / total_bin_volume)
- **Bins Used**: Number of bins containing items
- **Items Packed**: Number of successfully placed items
- **Success Rate**: Percentage of episodes where all items packed
- **Total Reward**: Cumulative reward from episode

## How It Works

The test scripts replicate the exact nesting process from `experiment_multi_problem.py`:

1. **Load model** → Load genome config + trained weights
2. **Load problem** → Parse problem file (bin dims, items, constraints)
3. **Create environment** → MultiBinPackingEnv with all constraints
4. **Run episodes**:
   - Enumerate feasible actions (placement options with constraint checks)
   - Build action features (25-dim) + heightmap patches (CNN input)
   - Agent selects action via neural network (greedy policy, ε=0)
   - Place item with gravity, update environment
   - Calculate reward
5. **Calculate metrics** → Utilization, bins used, success rate
6. **Generate visualizations** → 3D plots of packed bins (optional)

## Files Relationship

```
experiment_multi_problem.py         (Trains model)
         ↓
    trained_model.pth               (Model weights)
    evolved_genome.json             (Architecture)
         ↓
    test_model.py                   (Test single problem)
    test_model_batch.py             (Test multiple problems)
         ↓
    test_results/                   (Evaluation results)
    ├── test_results_*.json
    ├── batch_test_results.json
    ├── batch_test_summary.csv
    └── visualizations_*/
```

## Dependencies

Same as training scripts:
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib (for visualizations)
- Pandas (for batch summary)

## For More Details

See `TEST_README.md` for comprehensive documentation including:
- Detailed parameter explanations
- Output file formats
- Troubleshooting guide
- Performance tips
- Integration with experiments
