"""
The Graph API Client for fetching Uniswap V3 pool data

Port from JavaScript src/api/thegraph/ directory
"""
import httpx
import os
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta


# Subgraph IDs for Uniswap V3 on The Graph's decentralized network
# Source: https://docs.uniswap.org/api/subgraph/overview
SUBGRAPH_IDS = {
    0: '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV',  # Ethereum Mainnet
    1: 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj',  # Optimism
    2: 'FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM',  # Arbitrum
    3: '3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm',  # Polygon
    4: 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj',  # Perpetual (using Optimism)
    5: 'ESdrTJ3twMwWVoQ1hUE2u7PugEHX3QkenudD6aXCkDQ4',  # Celo
}

# Fallback URLs (deprecated hosted service, may stop working)
FALLBACK_URLS = {
    0: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
    1: "https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis",
    2: "https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal",
    3: "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon",
    5: "https://api.thegraph.com/subgraphs/name/jesse-sawa/uniswap-celo",
}


def url_for_protocol(protocol_id: int) -> str:
    """
    Get The Graph API URL for a given protocol.

    Args:
        protocol_id: Protocol identifier (0=Ethereum, 1=Optimism, etc.)

    Returns:
        The Graph API URL
    """
    api_key = os.getenv('GRAPH_API_KEY')

    if not api_key:
        print("Warning: GRAPH_API_KEY not set. Using fallback hosted service (deprecated).")
        print("Get your API key from: https://thegraph.com/studio/")
        return FALLBACK_URLS.get(protocol_id, FALLBACK_URLS[0])

    subgraph_id = SUBGRAPH_IDS.get(protocol_id, SUBGRAPH_IDS[0])
    return f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"


async def fetch_pool_hour_data(pool_id: str, from_date: int, protocol_id: int,
                               to_date: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch hourly pool data from The Graph (single query, max 1000 records).

    Port of getPoolHourData from uniPoolHourDatas.js

    Args:
        pool_id: Pool ID
        from_date: Unix timestamp to start from
        protocol_id: Protocol identifier
        to_date: Optional unix timestamp to end at (for pagination)

    Returns:
        List of hourly pool data records
    """
    # Build GraphQL query
    # NOTE: feeGrowthGlobal0X128, feeGrowthGlobal1X128, tick are required for whitepaper formulas
    if to_date:
        # Time-bounded query for pagination
        query = """
        query PoolHourDatas($pool: ID!, $fromdate: Int!, $todate: Int!) {
            poolHourDatas (
                where: { pool: $pool, periodStartUnix_gte: $fromdate, periodStartUnix_lt: $todate, close_gt: 0 },
                orderBy: periodStartUnix,
                orderDirection: asc,
                first: 1000
            ) {
                periodStartUnix
                liquidity
                high
                low
                open
                close
                volumeUSD
                tick
                feeGrowthGlobal0X128
                feeGrowthGlobal1X128
                pool {
                    id
                    totalValueLockedUSD
                    totalValueLockedToken1
                    totalValueLockedToken0
                    token0 { decimals }
                    token1 { decimals }
                    feeTier
                }
            }
        }
        """
        variables = {"pool": pool_id, "fromdate": from_date, "todate": to_date}
    else:
        # Most recent 1000 records after fromdate
        query = """
        query PoolHourDatas($pool: ID!, $fromdate: Int!) {
            poolHourDatas (
                where: { pool: $pool, periodStartUnix_gt: $fromdate, close_gt: 0 },
                orderBy: periodStartUnix,
                orderDirection: desc,
                first: 1000
            ) {
                periodStartUnix
                liquidity
                high
                low
                open
                close
                volumeUSD
                tick
                feeGrowthGlobal0X128
                feeGrowthGlobal1X128
                pool {
                    id
                    totalValueLockedUSD
                    totalValueLockedToken1
                    totalValueLockedToken0
                    token0 { decimals }
                    token1 { decimals }
                    feeTier
                }
            }
        }
        """
        variables = {"pool": pool_id, "fromdate": from_date}

    # Make request
    url = url_for_protocol(protocol_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                url,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            # Better error handling for response structure
            if not data:
                print(f"Empty response from The Graph API")
                return []

            if 'errors' in data:
                print(f"GraphQL errors: {data['errors']}")
                return []

            if 'data' not in data:
                print(f"No 'data' field in response: {data}")
                return []

            if data['data'] is None:
                print(f"'data' field is None in response")
                return []

            if 'poolHourDatas' not in data['data']:
                print(f"No 'poolHourDatas' in response data. Keys: {data['data'].keys() if data['data'] else 'None'}")
                return []

            return data['data']['poolHourDatas']

        except Exception as e:
            print(f"Error fetching pool hour data: {e}")
            import traceback
            traceback.print_exc()
            return []


async def fetch_pool_hour_data_paginated(pool_id: str, from_date: int,
                                         protocol_id: int,
                                         days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch hourly pool data for extended periods (beyond 41 days) using pagination.

    Port of getPoolHourDataPaginated from uniPoolHourDatasPaginated.js

    Args:
        pool_id: Pool ID
        from_date: Unix timestamp to start from
        protocol_id: Protocol identifier
        days: Number of days of data to fetch

    Returns:
        List of hourly pool data records, sorted by time (newest first)
    """
    CHUNK_SIZE_DAYS = 40  # 40 days = ~960 records (safe margin under 1000 limit)
    CHUNK_SIZE_SECONDS = CHUNK_SIZE_DAYS * 24 * 60 * 60

    chunks = []
    current_date = from_date
    now = int(datetime.now().timestamp())

    print(f"[Pagination] Fetching data from {datetime.fromtimestamp(from_date)} to {datetime.fromtimestamp(now)}")
    print(f"[Pagination] Days requested: {(now - from_date) // 86400} days")

    # Fetch data in chunks
    chunk_count = 0
    while current_date < now:
        chunk_end_date = min(current_date + CHUNK_SIZE_SECONDS, now)

        print(f"[Pagination] Chunk {chunk_count + 1}: {datetime.fromtimestamp(current_date)} to {datetime.fromtimestamp(chunk_end_date)}")

        try:
            # Fetch individual chunk with toDate limit
            chunk = await fetch_pool_hour_data(
                pool_id=pool_id,
                from_date=current_date,
                protocol_id=protocol_id,
                to_date=chunk_end_date
            )

            if not chunk:
                print(f"[Pagination] Empty chunk received for date range: {current_date} to {chunk_end_date}")
                # If first chunk fails, throw error
                if chunk_count == 0:
                    raise Exception("First chunk failed - cannot proceed")
                # Otherwise, use partial data
                break

            print(f"[Pagination] Chunk {chunk_count + 1} received {len(chunk)} records")

            # Safety check: if we're stuck in a loop (getting very few records repeatedly)
            if len(chunk) == 0:
                print("[Pagination] Empty chunk received, stopping pagination")
                break

            chunks.append(chunk)
            chunk_count += 1

            # Move to next chunk (no overlap needed since we use gte/lt which are non-overlapping)
            current_date = chunk_end_date

        except Exception as e:
            print(f"Error fetching chunk: {e}")
            # If first chunk fails, rethrow
            if chunk_count == 0:
                raise
            # Otherwise, break and use what we have
            break

    # Flatten all chunks
    all_data = []
    for chunk in chunks:
        all_data.extend(chunk)

    # Deduplicate and sort
    return deduplicate_and_sort(all_data)


def deduplicate_and_sort(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates records by periodStartUnix and sorts chronologically (descending - newest first).

    Port of deduplicateAndSort from uniPoolHourDatasPaginated.js

    Args:
        records: List of pool hour data records

    Returns:
        Deduplicated and sorted records (newest first)
    """
    unique_map = {}

    # Keep only the first occurrence of each timestamp
    for record in records:
        key = record['periodStartUnix']
        if key not in unique_map:
            unique_map[key] = record

    # Convert back to list and sort by timestamp (descending - newest first)
    # This matches the behavior of getPoolHourData which returns desc order
    sorted_records = sorted(
        unique_map.values(),
        key=lambda x: x['periodStartUnix'],
        reverse=True
    )

    return sorted_records


async def fetch_pool_info(pool_id: str, protocol_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch basic pool information.

    Args:
        pool_id: Pool ID
        protocol_id: Protocol identifier

    Returns:
        Pool info dict or None if not found
    """
    query = """
    query Pool($pool: ID!) {
        pool(id: $pool) {
            id
            token0 {
                id
                symbol
                decimals
            }
            token1 {
                id
                symbol
                decimals
            }
            feeTier
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
            tick
        }
    }
    """
    variables = {"pool": pool_id}

    url = url_for_protocol(protocol_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                url,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            if data and 'data' in data and 'pool' in data['data']:
                return data['data']['pool']
            else:
                return None

        except Exception as e:
            print(f"Error fetching pool info: {e}")
            return None


# Convenience function to fetch data for optimization
async def fetch_pool_data_for_optimization(pool_id: str, protocol_id: int,
                                          days: int = 30) -> Dict[str, Any]:
    """
    Fetch all necessary pool data for ML optimization.

    Args:
        pool_id: Pool ID
        protocol_id: Protocol identifier
        days: Number of days of historical data

    Returns:
        Dict with 'pool_info' and 'hourly_data'
    """
    # Calculate from_date
    from_date = int((datetime.now() - timedelta(days=days)).timestamp())

    # Fetch pool info and hourly data in parallel
    pool_info_task = fetch_pool_info(pool_id, protocol_id)

    # Use pagination if > 41 days
    if days > 41:
        hourly_data_task = fetch_pool_hour_data_paginated(pool_id, from_date, protocol_id, days)
    else:
        hourly_data_task = fetch_pool_hour_data(pool_id, from_date, protocol_id)

    pool_info, hourly_data = await asyncio.gather(pool_info_task, hourly_data_task)

    return {
        'pool_info': pool_info,
        'hourly_data': hourly_data
    }
