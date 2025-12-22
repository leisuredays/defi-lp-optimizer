"""
The Graph API 클라이언트

Uniswap V3 Subgraph에서 데이터를 조회하는 클라이언트.
다중 체인 지원 (Ethereum, Polygon, Optimism, Arbitrum, Celo)
"""

import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    import requests
except ImportError:
    requests = None  # Will raise error when used

from ..constants import SUBGRAPH_IDS, CHAIN_IDS
from .types import Pool, Tick, PoolHourData, Position
from . import queries


@dataclass
class GraphClientConfig:
    """Graph API 클라이언트 설정"""
    api_key: Optional[str] = None
    chain_id: int = 0  # Ethereum by default
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class GraphClientError(Exception):
    """Graph API 오류"""
    pass


class GraphClient:
    """The Graph API 클라이언트

    사용법:
        client = GraphClient(api_key="your_api_key", chain="ethereum")
        pool = client.get_pool("0x...")
        ticks = client.get_ticks("0x...", [tick_lower, tick_upper])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chain: str = "ethereum",
        timeout: int = 30
    ):
        """
        Args:
            api_key: The Graph API 키. None이면 환경변수에서 로드
            chain: 체인 이름 (ethereum, polygon, optimism, arbitrum, celo)
            timeout: 요청 타임아웃 (초)
        """
        if requests is None:
            raise ImportError("requests 패키지가 필요합니다: pip install requests")

        self.api_key = api_key or os.getenv("REACT_APP_GRAPH_API_KEY")
        if not self.api_key:
            raise GraphClientError(
                "API 키가 필요합니다. REACT_APP_GRAPH_API_KEY 환경변수를 설정하거나 "
                "api_key 파라미터로 전달하세요. "
                "API 키는 https://thegraph.com/studio/ 에서 발급받을 수 있습니다."
            )

        chain_lower = chain.lower()
        if chain_lower not in CHAIN_IDS:
            raise GraphClientError(
                f"지원하지 않는 체인: {chain}. "
                f"지원 체인: {', '.join(CHAIN_IDS.keys())}"
            )

        self.chain_id = CHAIN_IDS[chain_lower]
        self.subgraph_id = SUBGRAPH_IDS[self.chain_id]
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def endpoint(self) -> str:
        """GraphQL 엔드포인트 URL"""
        return f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/{self.subgraph_id}"

    def _execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """GraphQL 쿼리 실행

        Args:
            query: GraphQL 쿼리 문자열
            variables: 쿼리 변수
            max_retries: 최대 재시도 횟수

        Returns:
            쿼리 결과 데이터

        Raises:
            GraphClientError: API 오류 발생 시
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._session.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()

                if "errors" in data:
                    error_messages = [e.get("message", str(e)) for e in data["errors"]]
                    raise GraphClientError(f"GraphQL 오류: {'; '.join(error_messages)}")

                if "data" not in data:
                    raise GraphClientError("응답에 'data' 필드가 없습니다")

                return data["data"]

            except requests.exceptions.Timeout:
                last_error = GraphClientError(f"요청 타임아웃 ({self.timeout}초)")
            except requests.exceptions.RequestException as e:
                last_error = GraphClientError(f"네트워크 오류: {e}")
            except Exception as e:
                last_error = GraphClientError(f"예상치 못한 오류: {e}")

            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # Exponential backoff

        raise last_error

    def get_pool(self, pool_id: str) -> Optional[Pool]:
        """Pool 정보 조회

        Args:
            pool_id: Pool 컨트랙트 주소

        Returns:
            Pool 객체 또는 None
        """
        data = self._execute_query(queries.POOL_QUERY, {"id": pool_id.lower()})
        pool_data = data.get("pool")
        if not pool_data:
            return None
        return Pool.from_dict(pool_data)

    def get_ticks(self, pool_id: str, tick_idxs: List[int]) -> List[Tick]:
        """특정 틱들의 정보 조회

        수수료 계산을 위해 lower/upper 틱의 feeGrowthOutside 값 조회.

        Args:
            pool_id: Pool 컨트랙트 주소
            tick_idxs: 조회할 틱 인덱스 목록

        Returns:
            Tick 객체 목록
        """
        data = self._execute_query(
            queries.TICKS_BY_IDX_QUERY,
            {
                "pool": pool_id.lower(),
                "tickIdxs": [str(idx) for idx in tick_idxs]
            }
        )
        ticks_data = data.get("ticks", [])
        return [Tick.from_dict(t) for t in ticks_data]

    def get_all_ticks(
        self,
        pool_id: str,
        skip: int = 0,
        first: int = 1000
    ) -> List[Tick]:
        """Pool의 모든 틱 조회 (페이지네이션)

        Args:
            pool_id: Pool 컨트랙트 주소
            skip: 건너뛸 레코드 수
            first: 가져올 레코드 수 (최대 1000)

        Returns:
            Tick 객체 목록
        """
        data = self._execute_query(
            queries.ALL_TICKS_QUERY,
            {
                "pool": pool_id.lower(),
                "skip": skip,
                "first": min(first, 1000)
            }
        )
        ticks_data = data.get("ticks", [])
        return [Tick.from_dict(t) for t in ticks_data]

    def get_pool_hour_datas(
        self,
        pool_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[PoolHourData]:
        """시간별 Pool 데이터 조회

        Args:
            pool_id: Pool 컨트랙트 주소
            from_timestamp: 시작 Unix timestamp
            to_timestamp: 종료 Unix timestamp

        Returns:
            PoolHourData 객체 목록
        """
        data = self._execute_query(
            queries.POOL_HOUR_DATAS_QUERY,
            {
                "pool": pool_id.lower(),
                "fromdate": from_timestamp,
                "todate": to_timestamp
            }
        )
        hour_datas = data.get("poolHourDatas", [])
        return [PoolHourData.from_dict(d) for d in hour_datas]

    def get_pool_hour_datas_paginated(
        self,
        pool_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[PoolHourData]:
        """시간별 Pool 데이터 조회 (자동 페이지네이션)

        1000시간 이상의 데이터도 모두 조회합니다.
        갭이 있는 데이터도 끝까지 조회합니다.

        Args:
            pool_id: Pool 컨트랙트 주소
            from_timestamp: 시작 Unix timestamp
            to_timestamp: 종료 Unix timestamp

        Returns:
            PoolHourData 객체 목록 (시간순 정렬)
        """
        all_data: List[PoolHourData] = []
        current_from = from_timestamp
        hours_per_query = 1000
        seconds_per_hour = 3600

        while current_from < to_timestamp:
            current_to = min(
                current_from + (hours_per_query * seconds_per_hour),
                to_timestamp
            )

            batch = self.get_pool_hour_datas(pool_id, current_from, current_to)
            all_data.extend(batch)

            # 다음 배치의 시작점 결정
            if batch and len(batch) == hours_per_query:
                # 배치가 가득 찬 경우: 마지막 데이터 이후부터
                current_from = batch[-1].period_start_unix + seconds_per_hour
            else:
                # 배치가 비었거나 덜 찬 경우: 다음 시간 윈도우로 이동
                # (갭이 있을 수 있으므로 계속 진행)
                current_from = current_to

        return all_data

    def get_top_pools(
        self,
        first: int = 50,
        min_tvl: float = 10000
    ) -> List[Pool]:
        """TVL 기준 상위 풀 조회

        Args:
            first: 가져올 풀 수
            min_tvl: 최소 TVL (USD)

        Returns:
            Pool 객체 목록
        """
        data = self._execute_query(
            queries.TOP_POOLS_QUERY,
            {"first": first, "minTvl": str(min_tvl)}
        )
        pools_data = data.get("pools", [])
        return [Pool.from_dict(p) for p in pools_data]

    def get_fee_calculation_data(
        self,
        pool_id: str,
        tick_lower: int,
        tick_upper: int
    ) -> Dict[str, Any]:
        """수수료 계산에 필요한 모든 데이터 조회

        백서 Section 6.3, 6.4 기반 수수료 계산을 위한 9개 변수:
        - Global: f_g,0, f_g,1, i_c
        - Lower tick: f_o,0(i_l), f_o,1(i_l)
        - Upper tick: f_o,0(i_u), f_o,1(i_u)

        Args:
            pool_id: Pool 컨트랙트 주소
            tick_lower: 하한 틱
            tick_upper: 상한 틱

        Returns:
            {
                "pool": Pool,
                "tick_lower": Tick,
                "tick_upper": Tick
            }
        """
        pool = self.get_pool(pool_id)
        if not pool:
            raise GraphClientError(f"Pool을 찾을 수 없습니다: {pool_id}")

        ticks = self.get_ticks(pool_id, [tick_lower, tick_upper])
        ticks_by_idx = {t.tick_idx: t for t in ticks}

        lower_tick = ticks_by_idx.get(tick_lower)
        upper_tick = ticks_by_idx.get(tick_upper)

        if not lower_tick:
            # 틱이 초기화되지 않은 경우 기본값 사용
            lower_tick = Tick(
                tick_idx=tick_lower,
                liquidity_gross=0,
                liquidity_net=0,
                fee_growth_outside_0_x128=0,
                fee_growth_outside_1_x128=0
            )

        if not upper_tick:
            upper_tick = Tick(
                tick_idx=tick_upper,
                liquidity_gross=0,
                liquidity_net=0,
                fee_growth_outside_0_x128=0,
                fee_growth_outside_1_x128=0
            )

        return {
            "pool": pool,
            "tick_lower": lower_tick,
            "tick_upper": upper_tick
        }
