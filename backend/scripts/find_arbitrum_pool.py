"""
Find Arbitrum USDC/WETH 0.05% Pool

Simple script to find the specific pool for training.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.graph_client import GraphClient


async def find_arbitrum_usdc_weth_pool():
    """Find Arbitrum USDC/WETH 0.05% pool"""

    client = GraphClient()
    protocol_id = 2  # Arbitrum

    # Query for USDC/WETH pools on Arbitrum
    query = """
    query FindPool {
      pools(
        first: 20,
        orderBy: totalValueLockedUSD,
        orderDirection: desc,
        where: {
          feeTier: "500"
        }
      ) {
        id
        feeTier
        totalValueLockedUSD
        volumeUSD
        token0 {
          symbol
          decimals
        }
        token1 {
          symbol
          decimals
        }
        poolDayData(first: 1, orderBy: date, orderDirection: desc) {
          volumeUSD
        }
      }
    }
    """

    try:
        print("Searching for Arbitrum USDC/WETH 0.05% pool...")
        print("-" * 60)

        response = await client.query(protocol_id, query, {})
        pools = response.get('data', {}).get('pools', [])

        # Find USDC/WETH pool
        target_pool = None
        for pool in pools:
            token0 = pool['token0']['symbol']
            token1 = pool['token1']['symbol']

            # Check for USDC/WETH or WETH/USDC
            is_usdc_weth = (
                (token0.upper() in ['USDC', 'USDC.E'] and token1.upper() in ['WETH', 'WETH9']) or
                (token1.upper() in ['USDC', 'USDC.E'] and token0.upper() in ['WETH', 'WETH9'])
            )

            if is_usdc_weth:
                target_pool = pool
                break

        if target_pool:
            print(f"✓ Found pool!")
            print(f"\nPool Details:")
            print(f"  Pool ID: {target_pool['id']}")
            print(f"  Tokens: {target_pool['token0']['symbol']}/{target_pool['token1']['symbol']}")
            print(f"  Fee Tier: {int(target_pool['feeTier'])/10000}%")
            print(f"  TVL: ${float(target_pool['totalValueLockedUSD']):,.2f}")

            volume_24h = 0
            if target_pool.get('poolDayData') and len(target_pool['poolDayData']) > 0:
                volume_24h = float(target_pool['poolDayData'][0].get('volumeUSD', 0))
            print(f"  24h Volume: ${volume_24h:,.2f}")
            print(f"  Token0 decimals: {target_pool['token0']['decimals']}")
            print(f"  Token1 decimals: {target_pool['token1']['decimals']}")

            # Return pool config
            return {
                'pool_id': target_pool['id'],
                'protocol_id': 2,
                'protocol': 'arbitrum',
                'fee_tier': int(target_pool['feeTier']),
                'token0_symbol': target_pool['token0']['symbol'],
                'token1_symbol': target_pool['token1']['symbol'],
                'token0_decimals': int(target_pool['token0']['decimals']),
                'token1_decimals': int(target_pool['token1']['decimals']),
                'tvl': float(target_pool['totalValueLockedUSD']),
                'volume_24h': volume_24h
            }
        else:
            print("✗ USDC/WETH pool not found")
            print("\nAvailable 0.05% pools:")
            for pool in pools:
                print(f"  - {pool['token0']['symbol']}/{pool['token1']['symbol']}")
            return None

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    pool = asyncio.run(find_arbitrum_usdc_weth_pool())
    if pool:
        print(f"\n{'='*60}")
        print(f"Use this pool_id for data collection:")
        print(f"  {pool['pool_id']}")
        print(f"{'='*60}")
