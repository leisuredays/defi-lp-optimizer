"""
Pool Manager - Pool 데이터 관리

The Graph API에서 Pool 데이터를 조회하고 관리하는 클래스.
"""

from typing import Optional, List, Dict, Any

from ..data.types import Pool, Tick, PoolHourData
from ..data.graph_client import GraphClient
from ..math.tick_math import get_sqrt_ratio_at_tick, tick_to_price


class PoolManager:
    """Uniswap V3 Pool 관리자

    Pool 데이터 조회 및 캐싱을 담당합니다.

    사용법:
        manager = PoolManager(api_key="your_key", chain="ethereum")
        pool = manager.get_pool("0x...")
        ticks = manager.get_ticks_for_range("0x...", -100, 100)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chain: str = "ethereum"
    ):
        """
        Args:
            api_key: The Graph API 키
            chain: 체인 이름
        """
        self.client = GraphClient(api_key=api_key, chain=chain)
        self._pool_cache: Dict[str, Pool] = {}
        self._tick_cache: Dict[str, Dict[int, Tick]] = {}

    def get_pool(self, pool_id: str, use_cache: bool = True) -> Optional[Pool]:
        """Pool 정보 조회

        Args:
            pool_id: Pool 컨트랙트 주소
            use_cache: 캐시 사용 여부

        Returns:
            Pool 객체 또는 None
        """
        pool_id = pool_id.lower()

        if use_cache and pool_id in self._pool_cache:
            return self._pool_cache[pool_id]

        pool = self.client.get_pool(pool_id)
        if pool:
            self._pool_cache[pool_id] = pool

        return pool

    def get_ticks_for_range(
        self,
        pool_id: str,
        tick_lower: int,
        tick_upper: int,
        use_cache: bool = True
    ) -> Dict[int, Tick]:
        """범위 내 틱 정보 조회

        Args:
            pool_id: Pool 컨트랙트 주소
            tick_lower: 하한 틱
            tick_upper: 상한 틱
            use_cache: 캐시 사용 여부

        Returns:
            {tick_idx: Tick} 딕셔너리
        """
        pool_id = pool_id.lower()
        tick_idxs = [tick_lower, tick_upper]

        # 캐시 확인
        if use_cache and pool_id in self._tick_cache:
            cached = self._tick_cache[pool_id]
            if all(idx in cached for idx in tick_idxs):
                return {idx: cached[idx] for idx in tick_idxs}

        # API 조회
        ticks = self.client.get_ticks(pool_id, tick_idxs)
        ticks_by_idx = {t.tick_idx: t for t in ticks}

        # 캐시 저장
        if pool_id not in self._tick_cache:
            self._tick_cache[pool_id] = {}
        self._tick_cache[pool_id].update(ticks_by_idx)

        # 없는 틱은 기본값으로 생성
        for idx in tick_idxs:
            if idx not in ticks_by_idx:
                ticks_by_idx[idx] = Tick(
                    tick_idx=idx,
                    liquidity_gross=0,
                    liquidity_net=0,
                    fee_growth_outside_0_x128=0,
                    fee_growth_outside_1_x128=0
                )

        return ticks_by_idx

    def get_pool_hour_datas(
        self,
        pool_id: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[PoolHourData]:
        """시간별 Pool 데이터 조회 (자동 페이지네이션)

        Args:
            pool_id: Pool 컨트랙트 주소
            from_timestamp: 시작 Unix timestamp
            to_timestamp: 종료 Unix timestamp

        Returns:
            PoolHourData 목록
        """
        return self.client.get_pool_hour_datas_paginated(
            pool_id, from_timestamp, to_timestamp
        )

    def get_fee_calculation_data(
        self,
        pool_id: str,
        tick_lower: int,
        tick_upper: int
    ) -> Dict[str, Any]:
        """수수료 계산에 필요한 모든 데이터 조회

        Args:
            pool_id: Pool 컨트랙트 주소
            tick_lower: 하한 틱
            tick_upper: 상한 틱

        Returns:
            {pool, tick_lower, tick_upper} 딕셔너리
        """
        return self.client.get_fee_calculation_data(pool_id, tick_lower, tick_upper)

    def clear_cache(self):
        """캐시 초기화"""
        self._pool_cache.clear()
        self._tick_cache.clear()

    def get_current_price(self, pool_id: str, invert: bool = False) -> float:
        """현재 가격 조회

        Args:
            pool_id: Pool 컨트랙트 주소
            invert: True면 token0/token1, False면 token1/token0

        Returns:
            현재 가격
        """
        pool = self.get_pool(pool_id)
        if not pool:
            raise ValueError(f"Pool을 찾을 수 없습니다: {pool_id}")

        price = tick_to_price(
            pool.tick,
            pool.token0.decimals,
            pool.token1.decimals
        )

        return 1 / price if invert else price
