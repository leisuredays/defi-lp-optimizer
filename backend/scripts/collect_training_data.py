"""
Training Data Collection Script

Collects 6 months of hourly data from diverse Uniswap V3 pools for RL training.
"""
import asyncio
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.graph_client import GraphClient
from app.ml.pool_selector import PoolSelector


async def collect_pool_data(pool: dict, days: int = 180) -> pd.DataFrame:
    """
    Collect historical hourly data for a single pool.

    Args:
        pool: Pool metadata from PoolSelector
        days: Number of days to collect (default 180 = 6 months)

    Returns:
        DataFrame with hourly candle data
    """
    client = GraphClient()

    print(f"    Fetching {days} days of hourly data...")

    try:
        # Use paginated fetch for long history
        data = await client.fetch_pool_hour_data_paginated(
            pool_id=pool['pool_id'],
            from_date=days,
            protocol_id=pool['protocol_id'],
            days=days
        )

        if not data or len(data) == 0:
            raise ValueError("No data returned from The Graph")

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Data validation
        min_expected_hours = days * 20  # At least 20 hours per day
        if len(df) < min_expected_hours:
            print(f"    ⚠ Warning: Only {len(df)} hours (expected {min_expected_hours})")

        # Sort by timestamp
        df = df.sort_values('periodStartUnix', ascending=True)

        # Add pool metadata
        df['pool_id'] = pool['pool_id']
        df['protocol_id'] = pool['protocol_id']
        df['fee_tier'] = pool['fee_tier']
        df['token0_symbol'] = pool['token0_symbol']
        df['token1_symbol'] = pool['token1_symbol']
        df['token0_decimals'] = pool['token0_decimals']
        df['token1_decimals'] = pool['token1_decimals']

        # Convert numeric columns
        numeric_columns = ['close', 'high', 'low', 'open', 'volumeUSD',
                          'liquidity', 'feeGrowthGlobal0X128', 'feeGrowthGlobal1X128']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        raise Exception(f"Error collecting data: {e}")


async def main():
    """Main data collection pipeline"""

    print("="*70)
    print(" " * 15 + "TRAINING DATA COLLECTION")
    print("="*70)

    # Step 1: Select training pools
    print("\n[1/3] Selecting training pools...")
    print("-"*70)

    selector = PoolSelector()

    try:
        pools = await selector.select_training_pools(total_pools=20)
    except Exception as e:
        print(f"\n✗ Error selecting pools: {e}")
        return

    if len(pools) == 0:
        print("\n✗ No pools selected. Check your Graph API connection.")
        return

    print(f"\n✓ Selected {len(pools)} pools")

    # Display pool summary
    print("\nPool Summary:")
    protocols_count = {}
    for pool in pools:
        protocol = pool['protocol']
        protocols_count[protocol] = protocols_count.get(protocol, 0) + 1

    for protocol, count in protocols_count.items():
        print(f"  - {protocol.capitalize()}: {count} pools")

    # Step 2: Create data directory
    data_dir = Path("backend/data/training")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Collect data for each pool
    print(f"\n[2/3] Collecting data (180 days per pool)...")
    print("-"*70)

    successful = 0
    failed = 0

    for i, pool in enumerate(pools):
        pool_name = f"{pool['token0_symbol']}/{pool['token1_symbol']}"
        print(f"\n[{i+1}/{len(pools)}] {pool_name} ({pool['protocol']}, "
              f"{pool['fee_tier']/10000:.2f}%)")

        try:
            # Collect data
            df = await collect_pool_data(pool, days=180)

            # Save as parquet
            pool_id_short = pool['pool_id'][:8]
            output_file = data_dir / f"pool_{pool_id_short}_data.parquet"

            df.to_parquet(output_file, index=False, compression='snappy')

            # Display stats
            date_range = pd.to_datetime(df['periodStartUnix'], unit='s')
            print(f"    ✓ Saved {len(df)} records")
            print(f"      Date range: {date_range.min()} to {date_range.max()}")
            print(f"      File: {output_file.name}")

            successful += 1

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            failed += 1
            continue

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # Step 4: Summary
    print(f"\n[3/3] Collection Summary")
    print("-"*70)
    print(f"  Successful: {successful}/{len(pools)}")
    print(f"  Failed: {failed}/{len(pools)}")
    print(f"  Data directory: {data_dir.absolute()}")

    if successful > 0:
        print(f"\n✅ Data collection complete!")
        print(f"\nNext steps:")
        print(f"  1. Validate data: python scripts/validate_training_data.py")
        print(f"  2. Start training: python scripts/train_ppo.py")
    else:
        print(f"\n✗ No data collected successfully")

    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
