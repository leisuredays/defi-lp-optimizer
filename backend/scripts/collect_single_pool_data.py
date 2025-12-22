"""
Single Pool Data Collection

Collects 6 months of data for Arbitrum USDC/WETH 0.05% pool.
"""
import asyncio
import sys
from pathlib import Path
import pandas as pd
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core import graph_client


# Ethereum USDC/WETH 0.05% pool configuration
# Using Ethereum mainnet for full feeGrowthGlobal data support
POOL_CONFIG = {
    'pool_id': None,  # Will be auto-detected
    'protocol_id': 0,  # Ethereum Mainnet (has feeGrowthGlobal in PoolHourData)
    'protocol': 'ethereum',
    'fee_tier': 500,
    'days': 730  # 2 years (maximum available)
}


async def find_pool():
    """Auto-detect Ethereum USDC/WETH 0.05% pool"""

    query = """
    query FindPool {
      pools(
        first: 20,
        orderBy: totalValueLockedUSD,
        orderDirection: desc,
        where: { feeTier: "500" }
      ) {
        id
        feeTier
        totalValueLockedUSD
        token0 { symbol, decimals }
        token1 { symbol, decimals }
      }
    }
    """

    # Use graph_client module functions directly
    url = graph_client.url_for_protocol(POOL_CONFIG['protocol_id'])

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json={'query': query, 'variables': {}},
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()

    pools = data.get('data', {}).get('pools', [])

    # Find USDC/WETH pool
    for pool in pools:
        token0 = pool['token0']['symbol'].upper()
        token1 = pool['token1']['symbol'].upper()

        is_match = (
            (token0 in ['USDC', 'USDC.E'] and token1 in ['WETH', 'WETH9']) or
            (token1 in ['USDC', 'USDC.E'] and token0 in ['WETH', 'WETH9'])
        )

        if is_match:
            return {
                'pool_id': pool['id'],
                'token0_symbol': pool['token0']['symbol'],
                'token1_symbol': pool['token1']['symbol'],
                'token0_decimals': int(pool['token0']['decimals']),
                'token1_decimals': int(pool['token1']['decimals']),
                'tvl': float(pool['totalValueLockedUSD'])
            }

    raise ValueError("Ethereum USDC/WETH 0.05% pool not found")


async def collect_pool_data(pool: dict, days: int = 180):
    """Collect historical data for the pool"""

    print(f"    Fetching {days} days of hourly data...")

    # Calculate from_date (days ago from now)
    from datetime import datetime, timedelta
    from_date = int((datetime.now() - timedelta(days=days)).timestamp())

    data = await graph_client.fetch_pool_hour_data_paginated(
        pool_id=pool['pool_id'],
        from_date=from_date,
        protocol_id=POOL_CONFIG['protocol_id'],
        days=days
    )

    if not data or len(data) == 0:
        raise ValueError("No data returned")

    df = pd.DataFrame(data)

    # Validation
    min_hours = days * 20
    if len(df) < min_hours:
        print(f"    ⚠ Warning: Only {len(df)} hours (expected {min_hours})")

    # Sort by timestamp
    df = df.sort_values('periodStartUnix', ascending=True)

    # Add metadata
    df['pool_id'] = pool['pool_id']
    df['protocol_id'] = POOL_CONFIG['protocol_id']
    df['fee_tier'] = POOL_CONFIG['fee_tier']
    df['token0_symbol'] = pool['token0_symbol']
    df['token1_symbol'] = pool['token1_symbol']
    df['token0_decimals'] = pool['token0_decimals']
    df['token1_decimals'] = pool['token1_decimals']

    # Convert numeric columns (including tick for whitepaper formulas)
    numeric_cols = ['close', 'high', 'low', 'open', 'volumeUSD',
                   'liquidity', 'tick', 'feeGrowthGlobal0X128', 'feeGrowthGlobal1X128']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Verify required fields for whitepaper formulas
    required_fields = ['feeGrowthGlobal0X128', 'feeGrowthGlobal1X128', 'tick']
    missing = [f for f in required_fields if f not in df.columns or df[f].isna().all()]
    if missing:
        print(f"    ⚠ Warning: Missing whitepaper formula fields: {missing}")

    return df


async def main():
    """Main pipeline"""

    print("="*70)
    print(" " * 10 + "ETHEREUM USDC/WETH 0.05% DATA COLLECTION")
    print("="*70)

    # Step 1: Find pool
    print("\n[1/3] Finding Ethereum USDC/WETH 0.05% pool...")
    print("-"*70)

    try:
        pool = await find_pool()
        print(f"✓ Found pool: {pool['pool_id']}")
        print(f"  Tokens: {pool['token0_symbol']}/{pool['token1_symbol']}")
        print(f"  TVL: ${pool['tvl']:,.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Step 2: Collect data
    print(f"\n[2/3] Collecting {POOL_CONFIG['days']} days of data...")
    print("-"*70)

    try:
        df = await collect_pool_data(pool, days=POOL_CONFIG['days'])

        # Date range
        timestamps = pd.to_datetime(df['periodStartUnix'], unit='s')
        print(f"✓ Collected {len(df)} hourly records")
        print(f"  Date range: {timestamps.min()} to {timestamps.max()}")
        print(f"  Days: {(timestamps.max() - timestamps.min()).days}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Save data
    print(f"\n[3/3] Saving data...")
    print("-"*70)

    data_dir = Path(__file__).parent.parent / "data" / "training"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use descriptive filename
    filename = f"ethereum_usdc_weth_005_data.parquet"
    output_file = data_dir / filename

    df.to_parquet(output_file, index=False, compression='snappy')

    print(f"✓ Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Display stats
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print("-"*70)
    print(f"  Pool: {pool['token0_symbol']}/{pool['token1_symbol']}")
    print(f"  Protocol: Ethereum Mainnet")
    print(f"  Fee tier: 0.05%")
    print(f"  Records: {len(df):,} hours")
    print(f"  Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
    print(f"  Avg 24h volume: ${df['volumeUSD'].mean():,.2f}")

    missing_pct = (df['close'].isnull().sum() / len(df)) * 100
    print(f"  Missing data: {missing_pct:.2f}%")

    print(f"\n✅ Data collection complete!")
    print(f"\nNext steps:")
    print(f"  1. Update training_config.yaml with model name")
    print(f"  2. Run: python scripts/train_ppo_single_pool.py")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
