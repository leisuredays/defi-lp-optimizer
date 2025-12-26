#!/usr/bin/env python
"""
Visualize fold model behavior on both training and test datasets.

Usage:
  python visualize_fold.py --fold 1                    # Both train & test
  python visualize_fold.py --fold 1 --dataset train    # Train only
  python visualize_fold.py --fold 1 --dataset test     # Test only
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Configuration
WINDOW_DAYS = 180  # Training window
TEST_DAYS = 30     # Test window
WINDOW_HOURS = WINDOW_DAYS * 24  # 4320
TEST_HOURS = TEST_DAYS * 24      # 720

BASE_DIR = Path(__file__).parent.parent / "models" / "rolling_wfe"
MODEL_DIR = BASE_DIR / "models"
VIS_DIR = BASE_DIR / "visualizations"
SCRIPT_PATH = Path(__file__).parent / "visualize_model_behavior.py"


def visualize_fold(fold: int, dataset: str = "both", model_name: str = None):
    """
    Visualize model behavior on training and/or test data.

    Args:
        fold: Fold number (1-38)
        dataset: "train", "test", or "both"
        model_name: Optional model filename (default: fold_XX.zip)
    """
    if model_name:
        model_path = MODEL_DIR / model_name
        # Extract prefix for output naming (e.g., fold_01_900k from fold_01_900k.zip)
        output_prefix = model_name.replace('.zip', '')
    else:
        model_path = MODEL_DIR / f"fold_{fold:02d}.zip"
        output_prefix = f"fold_{fold:02d}"

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Calculate data indices for this fold
    train_start = (fold - 1) * TEST_HOURS  # Fold 1: 0, Fold 2: 720, ...
    test_start = train_start + WINDOW_HOURS

    tasks = []

    if dataset in ["train", "both"]:
        tasks.append({
            "name": "Train",
            "start": train_start,
            "length": WINDOW_HOURS,
            "output": VIS_DIR / f"{output_prefix}_train.png"
        })

    if dataset in ["test", "both"]:
        tasks.append({
            "name": "Test",
            "start": test_start,
            "length": TEST_HOURS,
            "output": VIS_DIR / f"{output_prefix}_test.png"
        })

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Visualizing Fold {fold} {task['name']} Set")
        print(f"{'='*60}")
        print(f"  Start index: {task['start']}")
        print(f"  Length: {task['length']} hours ({task['length']/24:.0f} days)")
        print(f"  Output: {task['output'].name}")
        print()

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--model", str(model_path),
            "--start", str(task['start']),
            "--length", str(task['length']),
            "--save", str(task['output'])
        ]

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"\n✅ Saved: {task['output']}")
        else:
            print(f"\n❌ Failed to generate {task['name']} visualization")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize fold model behavior',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_fold.py --fold 1                    # Both train & test
  python visualize_fold.py --fold 1 --dataset train    # Train only
  python visualize_fold.py --fold 1 --dataset test     # Test only
  python visualize_fold.py --fold 1 --model fold_01_900k.zip  # Specific model
        """
    )
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-38)')
    parser.add_argument('--dataset', choices=['train', 'test', 'both'],
                        default='both', help='Dataset to visualize')
    parser.add_argument('--model', type=str, default=None,
                        help='Model filename (default: fold_XX.zip)')

    args = parser.parse_args()

    if args.fold < 1 or args.fold > 38:
        print("Error: Fold must be between 1 and 38")
        sys.exit(1)

    visualize_fold(args.fold, args.dataset, args.model)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
