"""
Pool Selector for Training Data Collection

Selects diverse pools across protocols and fee tiers for RL training.
"""
import asyncio
from typing import List, Dict, Any
from app.core.graph_client import GraphClient


class PoolSelector:
    """Select diverse pools for training"""

    POOL_CRITERIA = {
        'ethereum': {'min_tvl': 1_000_000, 'fee_tiers': [500, 3000, 10000]},
        'optimism': {'min_tvl': 100_000, 'fee_tiers': [500, 3000, 10000]},
        'arbitrum': {'min_tvl': 100_000, 'fee_tiers': [500, 3000, 10000]},
        'polygon': {'min_tvl': 50_000, 'fee_tiers': [500, 3000, 10000]},
        'celo': {'min_tvl': 50_000, 'fee_tiers': [500, 3000, 10000]}
    }

    PROTOCOL_IDS = {
        'ethereum': 0,
        'optimism': 1,
        'arbitrum': 2,
        'polygon': 3,
        'celo': 5
    }

    def __init__(self):
        self.client = GraphClient()

    async def select_training_pools(self, total_pools: int = 20) -> List[Dict[str, Any]]:
        """
        Select diverse pools across protocols and fee tiers.

        Args:
            total_pools: Total number of pools to select (default 20)

        Returns:
            List of pool metadata:
            - pool_id: str
            - protocol_id: int
            - protocol: str (name)
            - fee_tier: int
            - tvl: float
            - volume_24h: float
            - token0_symbol: str
            - token1_symbol: str
            - token0_decimals: int
            - token1_decimals: int
        """
        pools_per_protocol = total_pools // len(self.PROTOCOL_IDS)
        selected_pools = []

        for protocol_name, protocol_id in self.PROTOCOL_IDS.items():
            print(f"\nQuerying {protocol_name.upper()} (protocol_id={protocol_id})...")

            try:
                # Fetch top pools for this protocol
                pools = await self._fetch_top_pools(protocol_id, protocol_name)

                # Select diverse pools (by fee tier)
                diverse_pools = self._select_diverse_pools(
                    pools,
                    n=pools_per_protocol,
                    protocol_name=protocol_name
                )

                selected_pools.extend(diverse_pools)

                print(f"  ✓ Selected {len(diverse_pools)} pools from {protocol_name}")

            except Exception as e:
                print(f"  ✗ Error querying {protocol_name}: {e}")
                continue

        return selected_pools[:total_pools]

    async def _fetch_top_pools(self, protocol_id: int, protocol_name: str,
                               limit: int = 50) -> List[Dict]:
        """
        Fetch top pools by TVL for a given protocol.

        Args:
            protocol_id: Protocol ID (0-5)
            protocol_name: Protocol name for criteria lookup
            limit: Number of pools to fetch

        Returns:
            List of pool data from The Graph
        """
        criteria = self.POOL_CRITERIA[protocol_name]

        # GraphQL query to fetch top pools
        query = """
        query TopPools($minTvl: String!) {
          pools(
            first: 50,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            where: { totalValueLockedUSD_gt: $minTvl }
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

        variables = {
            'minTvl': str(criteria['min_tvl'])
        }

        try:
            response = await self.client.query(protocol_id, query, variables)
            pools_data = response.get('data', {}).get('pools', [])

            # Parse and format pool data
            pools = []
            for pool in pools_data:
                # Get 24h volume
                volume_24h = 0
                if pool.get('poolDayData') and len(pool['poolDayData']) > 0:
                    volume_24h = float(pool['poolDayData'][0].get('volumeUSD', 0))

                pools.append({
                    'pool_id': pool['id'],
                    'protocol_id': protocol_id,
                    'protocol': protocol_name,
                    'fee_tier': int(pool['feeTier']),
                    'tvl': float(pool['totalValueLockedUSD']),
                    'volume_24h': volume_24h,
                    'token0_symbol': pool['token0']['symbol'],
                    'token1_symbol': pool['token1']['symbol'],
                    'token0_decimals': int(pool['token0']['decimals']),
                    'token1_decimals': int(pool['token1']['decimals'])
                })

            return pools

        except Exception as e:
            print(f"    Error fetching pools: {e}")
            return []

    def _select_diverse_pools(self, pools: List[Dict], n: int,
                             protocol_name: str) -> List[Dict]:
        """
        Select diverse pools by fee tier and volume.

        Args:
            pools: List of pool data
            n: Number of pools to select
            protocol_name: Protocol name for criteria

        Returns:
            Selected diverse pools
        """
        if len(pools) == 0:
            return []

        criteria = self.POOL_CRITERIA[protocol_name]
        fee_tiers = criteria['fee_tiers']

        # Group pools by fee tier
        pools_by_tier = {tier: [] for tier in fee_tiers}
        for pool in pools:
            if pool['fee_tier'] in fee_tiers:
                pools_by_tier[pool['fee_tier']].append(pool)

        # Sort each tier by volume (descending)
        for tier in pools_by_tier:
            pools_by_tier[tier].sort(key=lambda p: p['volume_24h'], reverse=True)

        # Select pools evenly from each tier
        selected = []
        pools_per_tier = n // len(fee_tiers)
        remainder = n % len(fee_tiers)

        for i, tier in enumerate(fee_tiers):
            tier_pools = pools_by_tier[tier]
            # Add 1 extra pool to first tiers if there's remainder
            count = pools_per_tier + (1 if i < remainder else 0)
            selected.extend(tier_pools[:count])

        # If we don't have enough, fill with top volume pools
        if len(selected) < n:
            all_sorted = sorted(pools, key=lambda p: p['volume_24h'], reverse=True)
            for pool in all_sorted:
                if pool not in selected:
                    selected.append(pool)
                    if len(selected) >= n:
                        break

        return selected[:n]


# Test function
async def test_pool_selector():
    """Test the pool selector"""
    selector = PoolSelector()

    print("="*60)
    print("Testing Pool Selector")
    print("="*60)

    try:
        pools = await selector.select_training_pools(total_pools=20)

        print(f"\n{'='*60}")
        print(f"Selected {len(pools)} pools:")
        print(f"{'='*60}\n")

        # Group by protocol for display
        by_protocol = {}
        for pool in pools:
            protocol = pool['protocol']
            if protocol not in by_protocol:
                by_protocol[protocol] = []
            by_protocol[protocol].append(pool)

        for protocol, protocol_pools in by_protocol.items():
            print(f"\n{protocol.upper()} ({len(protocol_pools)} pools):")
            for pool in protocol_pools:
                print(f"  - {pool['token0_symbol']}/{pool['token1_symbol']} "
                      f"(fee: {pool['fee_tier']/10000:.2f}%, "
                      f"TVL: ${pool['tvl']:,.0f}, "
                      f"Vol: ${pool['volume_24h']:,.0f})")

        print(f"\n{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_pool_selector())
