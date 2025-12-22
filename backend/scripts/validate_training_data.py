"""
Training Data Validation Script

Validates collected pool data for completeness and quality.
"""
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_pool_data(file_path: Path) -> Dict[str, Any]:
    """
    Validate a single pool data file.

    Args:
        file_path: Path to parquet file

    Returns:
        Dictionary with validation results
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return {
            'file': file_path.name,
            'valid': False,
            'errors': [f"Failed to read file: {e}"]
        }

    stats = {
        'file': file_path.name,
        'records': len(df),
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Get date range
    if 'periodStartUnix' in df.columns:
        timestamps = pd.to_datetime(df['periodStartUnix'], unit='s')
        stats['date_range'] = (
            timestamps.min().strftime('%Y-%m-%d'),
            timestamps.max().strftime('%Y-%m-%d')
        )
        stats['days_coverage'] = (timestamps.max() - timestamps.min()).days
    else:
        stats['valid'] = False
        stats['errors'].append("Missing 'periodStartUnix' column")

    # Check record count (minimum 150 days × 24 hours = 3600)
    MIN_RECORDS = 3600
    if len(df) < MIN_RECORDS:
        stats['valid'] = False
        stats['errors'].append(f"Insufficient records: {len(df)} < {MIN_RECORDS}")

    # Check required columns
    required_columns = ['close', 'high', 'low', 'periodStartUnix',
                       'feeGrowthGlobal0X128', 'feeGrowthGlobal1X128']

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        stats['valid'] = False
        stats['errors'].append(f"Missing columns: {missing_cols}")

    # Check for missing values in critical columns
    if 'close' in df.columns:
        missing_prices = df['close'].isnull().sum()
        missing_pct = (missing_prices / len(df)) * 100

        if missing_pct > 5:  # More than 5% missing
            stats['valid'] = False
            stats['errors'].append(
                f"Too many missing prices: {missing_prices} ({missing_pct:.1f}%)"
            )
        elif missing_pct > 1:  # 1-5% missing (warning)
            stats['warnings'].append(
                f"Some missing prices: {missing_prices} ({missing_pct:.1f}%)"
            )

    # Check for invalid prices
    if 'close' in df.columns:
        invalid_prices = (df['close'] <= 0).sum()
        if invalid_prices > 0:
            stats['valid'] = False
            stats['errors'].append(f"Invalid prices (≤0): {invalid_prices}")

    # Check price range
    if 'close' in df.columns and not df['close'].isnull().all():
        stats['price_range'] = (
            f"{df['close'].min():.6f}",
            f"{df['close'].max():.6f}"
        )

    # Check zero volume hours
    if 'volumeUSD' in df.columns:
        zero_volume = (df['volumeUSD'] == 0).sum()
        zero_volume_pct = (zero_volume / len(df)) * 100

        if zero_volume_pct > 50:  # More than 50% zero volume
            stats['warnings'].append(
                f"High zero-volume hours: {zero_volume} ({zero_volume_pct:.1f}%)"
            )

    # Check for gaps in time series
    if 'periodStartUnix' in df.columns:
        df_sorted = df.sort_values('periodStartUnix')
        time_diffs = df_sorted['periodStartUnix'].diff()

        # Expected: 3600 seconds (1 hour)
        gaps = time_diffs[time_diffs > 7200]  # Gaps > 2 hours
        if len(gaps) > 0:
            stats['warnings'].append(f"Time series gaps detected: {len(gaps)} gaps")

    return stats


def main():
    """Main validation pipeline"""

    print("="*70)
    print(" " * 20 + "DATA VALIDATION")
    print("="*70)

    # Find training data files
    data_dir = Path("backend/data/training")

    if not data_dir.exists():
        print(f"\n✗ Data directory not found: {data_dir}")
        print(f"  Run collect_training_data.py first")
        return

    files = sorted(data_dir.glob("pool_*_data.parquet"))

    if len(files) == 0:
        print(f"\n✗ No data files found in {data_dir}")
        print(f"  Run collect_training_data.py first")
        return

    print(f"\nValidating {len(files)} pool datasets...")
    print("-"*70)

    valid_count = 0
    warning_count = 0
    error_count = 0

    all_stats = []

    for file in files:
        stats = validate_pool_data(file)
        all_stats.append(stats)

        # Display status
        if stats['valid']:
            if stats['warnings']:
                status = "⚠"
                warning_count += 1
            else:
                status = "✓"
                valid_count += 1
        else:
            status = "✗"
            error_count += 1

        print(f"\n{status} {stats['file']}")
        print(f"   Records: {stats['records']:,}")

        if 'date_range' in stats:
            print(f"   Date range: {stats['date_range'][0]} to {stats['date_range'][1]} "
                  f"({stats['days_coverage']} days)")

        if 'price_range' in stats:
            print(f"   Price range: {stats['price_range'][0]} - {stats['price_range'][1]}")

        # Show errors
        for error in stats.get('errors', []):
            print(f"   ✗ ERROR: {error}")

        # Show warnings
        for warning in stats.get('warnings', []):
            print(f"   ⚠ WARNING: {warning}")

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print("-"*70)
    print(f"  Total datasets: {len(files)}")
    print(f"  ✓ Valid: {valid_count}")
    print(f"  ⚠ Valid with warnings: {warning_count}")
    print(f"  ✗ Invalid: {error_count}")

    if valid_count + warning_count >= 16:  # Need at least 16 for 80/10/10 split
        print(f"\n✅ Sufficient valid datasets for training!")
        print(f"\nNext step:")
        print(f"  python scripts/train_ppo.py")
    elif error_count > 0:
        print(f"\n⚠ Some datasets failed validation")
        print(f"  Consider re-collecting failed pools or proceeding with {valid_count + warning_count} pools")
    else:
        print(f"\n✗ Insufficient valid datasets for training (need at least 16)")
        print(f"  Re-run collect_training_data.py to gather more data")

    print("="*70)


if __name__ == "__main__":
    main()
