"""
GraphQL 쿼리 정의

The Graph API를 통해 Uniswap V3 데이터를 조회하기 위한 쿼리들.
백서 기반 정확한 수수료 계산을 위해 필요한 모든 필드 포함.
"""

# Pool 정보 쿼리 (Global State)
POOL_QUERY = """
query Pool($id: ID!) {
  pool(id: $id) {
    id
    feeTier
    tick
    sqrtPrice
    liquidity
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    token0 {
      id
      symbol
      name
      decimals
    }
    token1 {
      id
      symbol
      name
      decimals
    }
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
  }
}
"""

# 특정 틱들의 정보 쿼리 (Tick-Indexed State)
# feeGrowthOutside 필드 포함 - 백서 Section 6.3 수수료 계산에 필수
TICKS_BY_IDX_QUERY = """
query Ticks($pool: ID!, $tickIdxs: [BigInt!]!) {
  ticks(where: { pool: $pool, tickIdx_in: $tickIdxs }) {
    tickIdx
    liquidityGross
    liquidityNet
    feeGrowthOutside0X128
    feeGrowthOutside1X128
  }
}
"""

# Pool의 모든 틱 조회 (유동성 분포 시각화용)
ALL_TICKS_QUERY = """
query AllTicks($pool: ID!, $skip: Int!, $first: Int!) {
  ticks(
    where: { pool: $pool }
    orderBy: tickIdx
    orderDirection: asc
    skip: $skip
    first: $first
  ) {
    tickIdx
    liquidityGross
    liquidityNet
    feeGrowthOutside0X128
    feeGrowthOutside1X128
  }
}
"""

# 시간별 Pool 데이터 쿼리 (백테스트용)
POOL_HOUR_DATAS_QUERY = """
query PoolHourDatas($pool: ID!, $fromdate: Int!, $todate: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gte: $fromdate, periodStartUnix_lt: $todate, close_gt: 0 }
    orderBy: periodStartUnix
    orderDirection: asc
    first: 1000
  ) {
    periodStartUnix
    tick
    sqrtPrice
    liquidity
    high
    low
    close
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    pool {
      token0 { decimals }
      token1 { decimals }
    }
  }
}
"""

# 최근 시간별 데이터 쿼리 (페이지네이션 없이 최근 1000개)
RECENT_POOL_HOUR_DATAS_QUERY = """
query RecentPoolHourDatas($pool: ID!, $fromdate: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gt: $fromdate, close_gt: 0 }
    orderBy: periodStartUnix
    orderDirection: desc
    first: 1000
  ) {
    periodStartUnix
    tick
    sqrtPrice
    liquidity
    high
    low
    close
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    pool {
      token0 { decimals }
      token1 { decimals }
    }
  }
}
"""

# 일별 Pool 데이터 쿼리
POOL_DAY_DATAS_QUERY = """
query PoolDayDatas($pool: ID!, $fromdate: Int!, $todate: Int!) {
  poolDayDatas(
    where: { pool: $pool, date_gte: $fromdate, date_lt: $todate }
    orderBy: date
    orderDirection: asc
    first: 1000
  ) {
    date
    tick
    sqrtPrice
    liquidity
    high
    low
    close
    open
    volumeUSD
    feesUSD
    tvlUSD
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
  }
}
"""

# TVL 기준 상위 풀 조회
TOP_POOLS_QUERY = """
query TopPools($first: Int!, $minTvl: BigDecimal!) {
  pools(
    first: $first
    where: { totalValueLockedUSD_gt: $minTvl, volumeUSD_gt: 1 }
    orderBy: totalValueLockedUSD
    orderDirection: desc
  ) {
    id
    feeTier
    tick
    sqrtPrice
    liquidity
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    token0 {
      id
      symbol
      name
      decimals
    }
    token1 {
      id
      symbol
      name
      decimals
    }
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
  }
}
"""

# 토큰으로 풀 검색
POOLS_BY_TOKEN_QUERY = """
query PoolsByToken($token: ID!, $minTvl: BigDecimal!) {
  token0Pools: pools(
    where: { token0: $token, totalValueLockedUSD_gt: $minTvl }
    orderBy: totalValueLockedUSD
    orderDirection: desc
  ) {
    id
    feeTier
    tick
    sqrtPrice
    liquidity
    token0 { id, symbol, name, decimals }
    token1 { id, symbol, name, decimals }
    totalValueLockedUSD
  }
  token1Pools: pools(
    where: { token1: $token, totalValueLockedUSD_gt: $minTvl }
    orderBy: totalValueLockedUSD
    orderDirection: desc
  ) {
    id
    feeTier
    tick
    sqrtPrice
    liquidity
    token0 { id, symbol, name, decimals }
    token1 { id, symbol, name, decimals }
    totalValueLockedUSD
  }
}
"""

# Position 조회 (NonfungiblePositionManager 기반)
POSITION_QUERY = """
query Position($id: ID!) {
  position(id: $id) {
    id
    owner
    pool {
      id
      tick
      sqrtPrice
      feeGrowthGlobal0X128
      feeGrowthGlobal1X128
      token0 { id, symbol, decimals }
      token1 { id, symbol, decimals }
    }
    tickLower {
      tickIdx
      feeGrowthOutside0X128
      feeGrowthOutside1X128
    }
    tickUpper {
      tickIdx
      feeGrowthOutside0X128
      feeGrowthOutside1X128
    }
    liquidity
    depositedToken0
    depositedToken1
    feeGrowthInside0LastX128
    feeGrowthInside1LastX128
    collectedFeesToken0
    collectedFeesToken1
  }
}
"""
