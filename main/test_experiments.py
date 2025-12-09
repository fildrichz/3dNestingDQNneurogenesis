"""
Quick test script to verify experiment implementations.

This runs minimal experiments (small population, few generations) to ensure:
1. Code runs without errors
2. Checkpointing works correctly
3. Results are saved properly

For full experiments, use the main experiment scripts with proper parameters.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Test configuration
TEST_DATASET_DIR = "nesting/inputData/Benchmark dataset and instance generator for Real-World 3dBPP/Input"
TEST_RESULTS_DIR = "test_results"

# Small parameters for quick testing
TEST_CONFIG = {
    'population_size': 100,
    'generations': 100,
    'episodes_per_eval': 200,
    'training_episodes': 1000,
    'elite_size': 10,
    'mutation_rate': 0.2,
    'seed': 42,
    'verbose': False,
    'resume': False
}


def test_problem_specific():
    """Test Experiment 1: Problem-Specific Evolution"""
    print("\n" + "="*80)
    print("TEST 1: Problem-Specific Architecture Evolution")
    print("="*80 + "\n")

    from experiment_problem_specific import run_problem_specific_experiment

    # Clean test directory
    test_dir = Path(TEST_RESULTS_DIR) / "experiment1"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    try:
        # Run on just one problem for testing
        # Create a temporary dataset directory with only one problem
        temp_dataset = Path(TEST_RESULTS_DIR) / "temp_dataset"
        temp_dataset.mkdir(parents=True, exist_ok=True)

        # Copy one problem file
        dataset_path = Path(TEST_DATASET_DIR)
        test_problem = dataset_path / "3dBPP_1.txt"
        if test_problem.exists():
            shutil.copy(test_problem, temp_dataset / "3dBPP_1.txt")
        else:
            print(f"ERROR: Test problem not found at {test_problem}")
            return False

        # Run experiment
        run_problem_specific_experiment(
            dataset_dir=str(temp_dataset),
            results_dir=str(test_dir),
            **TEST_CONFIG
        )

        # Verify results
        assert test_dir.exists(), "Results directory not created"
        assert (test_dir / "experiment_state.json").exists(), "State file not created"
        assert (test_dir / "experiment_summary.json").exists(), "Summary not created"
        assert (test_dir / "3dBPP_1").exists(), "Problem directory not created"
        assert (test_dir / "3dBPP_1" / "evolved_genome.json").exists(), "Genome not saved"
        assert (test_dir / "3dBPP_1" / "trained_model.pth").exists(), "Model not saved"

        # Check checkpoint
        checkpoint = test_dir / "3dBPP_1" / "evolution" / "checkpoint_latest.json"
        assert checkpoint.exists(), "Checkpoint not created"

        with open(checkpoint, 'r') as f:
            data = json.load(f)
            assert data['generation'] == TEST_CONFIG['generations'], "Wrong generation in checkpoint"

        print("\n" + "="*80)
        print("TEST 1 PASSED: Problem-specific experiment works correctly")
        print("="*80 + "\n")

        # Cleanup temp dataset
        shutil.rmtree(temp_dataset)

        return True

    except Exception as e:
        print(f"\nTEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_leave_one_out():
    """Test Experiment 2: Leave-One-Out"""
    print("\n" + "="*80)
    print("TEST 2: Leave-One-Out Generalization")
    print("="*80 + "\n")

    from experiment_leave_one_out import run_leave_one_out_experiment

    # Clean test directory
    test_dir = Path(TEST_RESULTS_DIR) / "experiment2"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    try:
        # Create temporary dataset with 3 problems
        temp_dataset = Path(TEST_RESULTS_DIR) / "temp_dataset_loo"
        temp_dataset.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(TEST_DATASET_DIR)
        for i in [1, 2, 3]:
            problem_file = dataset_path / f"3dBPP_{i}.txt"
            if problem_file.exists():
                shutil.copy(problem_file, temp_dataset / f"3dBPP_{i}.txt")

        # Run experiment (hold out 3dBPP_3, train on 3dBPP_1 and 3dBPP_2)
        run_leave_one_out_experiment(
            dataset_dir=str(temp_dataset),
            target_problem="3dBPP_3",
            results_dir=str(test_dir),
            episodes_per_problem=5,  # Even fewer for multi-problem
            **{k: v for k, v in TEST_CONFIG.items() if k != 'episodes_per_eval'}
        )

        # Verify results
        assert test_dir.exists(), "Results directory not created"
        assert (test_dir / "results.json").exists(), "Results not created"
        assert (test_dir / "evolved_genome.json").exists(), "Genome not saved"
        assert (test_dir / "trained_model.pth").exists(), "Model not saved"

        # Check checkpoint
        checkpoint = test_dir / "evolution" / "checkpoint_latest.json"
        assert checkpoint.exists(), "Checkpoint not created"

        # Verify results content
        with open(test_dir / "results.json", 'r') as f:
            results = json.load(f)
            assert results['target_problem'] == "3dBPP_3", "Wrong target problem"
            assert len(results['training_problems']) == 2, "Wrong number of training problems"
            assert '3dBPP_1' in results['training_problems'], "Missing training problem"
            assert '3dBPP_2' in results['training_problems'], "Missing training problem"

        print("\n" + "="*80)
        print("TEST 2 PASSED: Leave-one-out experiment works correctly")
        print("="*80 + "\n")

        # Cleanup temp dataset
        shutil.rmtree(temp_dataset)

        return True

    except Exception as e:
        print(f"\nTEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_resume():
    """Test checkpoint and resume functionality"""
    print("\n" + "="*80)
    print("TEST 3: Checkpoint and Resume")
    print("="*80 + "\n")

    from experiment_problem_specific import run_problem_specific_experiment

    test_dir = Path(TEST_RESULTS_DIR) / "experiment_resume"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    try:
        # Create temp dataset
        temp_dataset = Path(TEST_RESULTS_DIR) / "temp_dataset_resume"
        temp_dataset.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(TEST_DATASET_DIR)
        test_problem = dataset_path / "3dBPP_1.txt"
        if test_problem.exists():
            shutil.copy(test_problem, temp_dataset / "3dBPP_1.txt")

        # Run partial experiment (only 1 generation)
        print("Running partial experiment (1 generation)...")
        config = TEST_CONFIG.copy()
        config['generations'] = 1
        config['resume'] = False

        run_problem_specific_experiment(
            dataset_dir=str(temp_dataset),
            results_dir=str(test_dir),
            **config
        )

        # Verify checkpoint exists
        checkpoint_file = test_dir / "3dBPP_1" / "evolution" / "checkpoint_latest.json"
        assert checkpoint_file.exists(), "Checkpoint not created after first run"

        with open(checkpoint_file, 'r') as f:
            checkpoint1 = json.load(f)
            assert checkpoint1['generation'] == 1, "Wrong generation in first checkpoint"

        # Resume and run one more generation
        print("\nResuming experiment (1 more generation)...")
        config['generations'] = 2
        config['resume'] = True

        run_problem_specific_experiment(
            dataset_dir=str(temp_dataset),
            results_dir=str(test_dir),
            **config
        )

        # Verify checkpoint was updated
        with open(checkpoint_file, 'r') as f:
            checkpoint2 = json.load(f)
            assert checkpoint2['generation'] == 2, "Checkpoint not updated after resume"

        print("\n" + "="*80)
        print("TEST 3 PASSED: Checkpoint and resume works correctly")
        print("="*80 + "\n")

        # Cleanup
        shutil.rmtree(temp_dataset)

        return True

    except Exception as e:
        print(f"\nTEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("EXPERIMENT IMPLEMENTATION TESTS")
    print("="*80)
    print("\nThis will run minimal experiments to verify:")
    print("1. Problem-specific evolution works")
    print("2. Leave-one-out evolution works")
    print("3. Checkpoint/resume functionality works")
    print("\nUsing small parameters (3 genomes, 2 generations) for quick testing.")
    print("="*80 + "\n")

    # Create test results directory
    Path(TEST_RESULTS_DIR).mkdir(exist_ok=True)

    results = {
        'test_1_problem_specific': False,
        'test_2_leave_one_out': False,
        'test_3_checkpoint_resume': False
    }

    # Run tests
    results['test_1_problem_specific'] = test_problem_specific()
    results['test_2_leave_one_out'] = test_leave_one_out()
    results['test_3_checkpoint_resume'] = test_checkpoint_resume()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("="*80)
    if all_passed:
        print("\nALL TESTS PASSED - Implementation is working correctly!")
        print("You can now run full experiments with proper parameters.")
    else:
        print("\nSOME TESTS FAILED - Please check errors above.")
    print("="*80 + "\n")

    # Cleanup test results
    cleanup = input("Delete test results directory? [y/N]: ")
    if cleanup.lower() == 'y':
        shutil.rmtree(TEST_RESULTS_DIR)
        print(f"Deleted {TEST_RESULTS_DIR}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
