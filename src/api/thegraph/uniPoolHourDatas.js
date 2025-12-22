import { urlForProtocol, requestBody } from "./helpers";

export const getPoolHourData = async (pool, fromdate, signal, protocol, todate = null) => {

  // When todate is provided, we're fetching a specific range (for pagination)
  // Otherwise, we're fetching the most recent 1000 records after fromdate
  const query = todate
    ? `query PoolHourDatas($pool: ID!, $fromdate: Int!, $todate: Int!) {
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
          pool {
            id
            totalValueLockedUSD
            totalValueLockedToken1
            totalValueLockedToken0
            token0 { decimals }
            token1 { decimals }
          }
          close
          feeGrowthGlobal0X128
          feeGrowthGlobal1X128
        }
      }`
    : `query PoolHourDatas($pool: ID!, $fromdate: Int!) {
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
          pool {
            id
            totalValueLockedUSD
            totalValueLockedToken1
            totalValueLockedToken0
            token0 { decimals }
            token1 { decimals }
          }
          close
          feeGrowthGlobal0X128
          feeGrowthGlobal1X128
        }
      }`;

  const variables = todate
    ? { pool: pool, fromdate: fromdate, todate: todate }
    : { pool: pool, fromdate: fromdate };

  const url = urlForProtocol(protocol);

  try {
    const response = await fetch(url, requestBody({query: query, variables: variables, signal: signal}));
    const data = await response.json();

    if (data && data.data && data.data.poolHourDatas) {
      return data.data.poolHourDatas;
    }
    else {
      console.log("nothing returned from getPoolHourData", data);
      return null;
    }

  } catch (error) {
    return {error: error};
  }

}