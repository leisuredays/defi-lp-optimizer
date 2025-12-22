"""
WETH/USDT 0.3% Pool Data Collection (Ethereum Mainnet)

Collects 2 years of hourly data for PPO model training.
Target pool: 0x4e68ccd3e89f51c3074ca5072bbac773960dfa36

Usage:
    source /home/zekiya/miniconda3/etc/profile.d/conda.sh && \
    conda activate uniswap-v3-backend && \
    python scripts/collect_weth_usdt_data.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# Add project root for uniswap_v3 module access
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use the uniswap_v3 module's GraphClient (proper API key support)
from uniswap_v3.data.graph_client import GraphClient


# WETH/USDT 0.3% pool configuration (Ethereum Mainnet)
POOL_CONFIG = {
    'pool_id': '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
    'protocol_id': 0,  # Ethereum Mainnet
    'protocol': 'ethereum',
    'fee_tier': 3000,  # 0.3%
    'days': 730,  # 2 years
    'token0_symbol': 'WETH',
    'token1_symbol': 'USDT',
    'token0_decimals': 18,
    'token1_decimals': 6,
}


def verify_pool(client: GraphClient):
    """Verify the pool exists and get current info"""

    pool = client.get_pool(POOL_CONFIG['pool_id'])

    if not pool:
        raise ValueError(f"Pool {POOL_CONFIG['pool_id']} not found")

    return {
        'pool_id': pool.id,
        'token0_symbol': pool.token0.symbol,
        'token1_symbol': pool.token1.symbol,
        'token0_decimals': pool.token0.decimals,
        'token1_decimals': pool.token1.decimals,
        'tvl': pool.total_value_locked_usd or 0,
        'fee_tier': pool.fee_tier
    }


def collect_pool_data(client: GraphClient, days: int = 730):
    """Collect historical data for WETH/USDT 0.3% pool"""

    print(f"    Fetching {days} days of hourly data...")

    from_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    hour_datas = client.get_pool_hour_datas_paginated(
        pool_id=POOL_CONFIG['pool_id'],
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp
    )

    if not hour_datas or len(hour_datas) == 0:
        raise ValueError("No data returned from The Graph")

    # Convert to list of dicts (using snake_case attributes from PoolHourData)
    data = []
    for hd in hour_datas:
        data.append({
            'periodStartUnix': hd.period_start_unix,
            'open': hd.close,  # PoolHourData doesn't have open, use close as proxy
            'high': hd.high,
            'low': hd.low,
            'close': hd.close,
            'volumeUSD': 0.0,  # Not available in PoolHourData
            'liquidity': hd.liquidity,
            'tick': hd.tick,
            'feeGrowthGlobal0X128': hd.fee_growth_global_0_x128,
            'feeGrowthGlobal1X128': hd.fee_growth_global_1_x128,
        })

    df = pd.DataFrame(data)

    # Sort by timestamp
    df = df.sort_values('periodStartUnix', ascending=True)

    # Add metadata
    df['pool_id'] = POOL_CONFIG['pool_id']
    df['protocol_id'] = POOL_CONFIG['protocol_id']
    df['fee_tier'] = POOL_CONFIG['fee_tier']
    df['token0_symbol'] = POOL_CONFIG['token0_symbol']
    df['token1_symbol'] = POOL_CONFIG['token1_symbol']
    df['token0_decimals'] = POOL_CONFIG['token0_decimals']
    df['token1_decimals'] = POOL_CONFIG['token1_decimals']

    # Convert numeric columns
    numeric_cols = ['close', 'high', 'low', 'open', 'volumeUSD',
                   'liquidity', 'tick', 'feeGrowthGlobal0X128', 'feeGrowthGlobal1X128']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Verify required fields for whitepaper formulas
    required_fields = ['feeGrowthGlobal0X128', 'feeGrowthGlobal1X128', 'tick']
    missing = [f for f in required_fields if f not in df.columns or df[f].isna().all()]
    if missing:
        print(f"    Warning: Missing whitepaper formula fields: {missing}")

    return df


def main():
    """Main pipeline"""

    print("="*70)
    print(" " * 10 + "WETH/USDT 0.3% DATA COLLECTION (ETHEREUM)")
    print("="*70)

    # Initialize client
    print("\nInitializing GraphClient...")
    try:
        client = GraphClient(chain="ethereum")
        print("  GraphClient initialized")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Step 1: Verify pool
    print(f"\n[1/3] Verifying pool: {POOL_CONFIG['pool_id'][:20]}...")
    print("-"*70)

    try:
        pool = verify_pool(client)
        print(f"  Pool ID: {pool['pool_id']}")
        print(f"  Tokens: {pool['token0_symbol']}/{pool['token1_symbol']}")
        print(f"  Decimals: {pool['token0_decimals']}/{pool['token1_decimals']}")
        print(f"  TVL: ${pool['tvl']:,.2f}")
        print(f"  Fee tier: {pool['fee_tier'] / 10000:.2f}%")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Step 2: Collect data
    print(f"\n[2/3] Collecting {POOL_CONFIG['days']} days of data...")
    print("-"*70)

    try:
        df = collect_pool_data(client, days=POOL_CONFIG['days'])

        timestamps = pd.to_datetime(df['periodStartUnix'], unit='s')
        print(f"  Collected {len(df)} hourly records")
        print(f"  Date range: {timestamps.min()} to {timestamps.max()}")
        print(f"  Days: {(timestamps.max() - timestamps.min()).days}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Save data
    print(f"\n[3/3] Saving data...")
    print("-"*70)

    data_dir = Path(__file__).parent.parent / "data" / "training"
    data_dir.mkdir(parents=True, exist_ok=True)

    filename = "ethereum_weth_usdt_03_data.parquet"
    output_file = data_dir / filename

    df.to_parquet(output_file, index=False, compression='snappy')

    print(f"  Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Display stats
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print("-"*70)
    print(f"  Pool: {POOL_CONFIG['token0_symbol']}/{POOL_CONFIG['token1_symbol']}")
    print(f"  Protocol: Ethereum Mainnet")
    print(f"  Fee tier: 0.3%")
    print(f"  Records: {len(df):,} hours")
    print(f"  Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")

    # Check feeGrowthGlobal availability
    fg0_available = df['feeGrowthGlobal0X128'].notna().sum()
    fg1_available = df['feeGrowthGlobal1X128'].notna().sum()
    print(f"  feeGrowthGlobal0X128: {fg0_available}/{len(df)} ({100*fg0_available/len(df):.1f}%)")
    print(f"  feeGrowthGlobal1X128: {fg1_available}/{len(df)} ({100*fg1_available/len(df):.1f}%)")

    missing_pct = (df['close'].isnull().sum() / len(df)) * 100
    print(f"  Missing price data: {missing_pct:.2f}%")

    print(f"\n  Data collection complete!")
    print(f"\n  Next steps:")
    print(f"  1. Run: python scripts/train_ppo_full_autonomy.py")
    print("="*70)


if __name__ == "__main__":
    main()
