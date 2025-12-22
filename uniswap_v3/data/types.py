"""
Uniswap V3 데이터 타입 정의

The Graph API에서 반환되는 데이터 구조를 Python dataclass로 정의.
모든 숫자 필드는 온체인 정밀도를 위해 int 타입 사용.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Token:
    """ERC20 토큰 정보"""
    id: str  # 컨트랙트 주소
    symbol: str
    name: str
    decimals: int

    @classmethod
    def from_dict(cls, data: dict) -> "Token":
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            name=data["name"],
            decimals=int(data["decimals"])
        )


@dataclass
class Pool:
    """Uniswap V3 Pool 정보

    Global State (Section 6.2, Table 1):
    - liquidity: 현재 가격에서 활성화된 총 유동성
    - sqrtPrice: 현재 √가격 (Q96 인코딩)
    - tick: 현재 틱 인덱스
    - feeGrowthGlobal0X128: token0 단위유동성당 누적수수료 (Q128)
    - feeGrowthGlobal1X128: token1 단위유동성당 누적수수료 (Q128)
    """
    id: str  # Pool 컨트랙트 주소
    fee_tier: int  # 수수료 티어 (100, 500, 3000, 10000 bps)
    tick: int  # 현재 틱 인덱스 (i_c)
    sqrt_price: int  # sqrtPriceX96
    liquidity: int  # 현재 활성 유동성 (L)
    fee_growth_global_0_x128: int  # f_g,0
    fee_growth_global_1_x128: int  # f_g,1
    token0: Token
    token1: Token
    total_value_locked_usd: Optional[float] = None
    total_value_locked_token0: Optional[float] = None
    total_value_locked_token1: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Pool":
        return cls(
            id=data["id"],
            fee_tier=int(data["feeTier"]),
            tick=int(data["tick"]),
            sqrt_price=int(data["sqrtPrice"]),
            liquidity=int(data["liquidity"]),
            fee_growth_global_0_x128=int(data.get("feeGrowthGlobal0X128", 0)),
            fee_growth_global_1_x128=int(data.get("feeGrowthGlobal1X128", 0)),
            token0=Token.from_dict(data["token0"]),
            token1=Token.from_dict(data["token1"]),
            total_value_locked_usd=float(data.get("totalValueLockedUSD", 0)) if data.get("totalValueLockedUSD") else None,
            total_value_locked_token0=float(data.get("totalValueLockedToken0", 0)) if data.get("totalValueLockedToken0") else None,
            total_value_locked_token1=float(data.get("totalValueLockedToken1", 0)) if data.get("totalValueLockedToken1") else None,
        )


@dataclass
class Tick:
    """Tick-Indexed State (Section 6.3, Table 2)

    - tickIdx: 틱 인덱스
    - liquidityGross: 해당 틱을 경계로 하는 총 유동성
    - liquidityNet: 틱 크로싱 시 유동성 변화량 (ΔL)
    - feeGrowthOutside0X128: 틱 외부 누적수수료 token0 (f_o,0)
    - feeGrowthOutside1X128: 틱 외부 누적수수료 token1 (f_o,1)
    """
    tick_idx: int
    liquidity_gross: int
    liquidity_net: int
    fee_growth_outside_0_x128: int  # f_o,0
    fee_growth_outside_1_x128: int  # f_o,1

    @classmethod
    def from_dict(cls, data: dict) -> "Tick":
        return cls(
            tick_idx=int(data["tickIdx"]),
            liquidity_gross=int(data.get("liquidityGross", 0)),
            liquidity_net=int(data.get("liquidityNet", 0)),
            fee_growth_outside_0_x128=int(data.get("feeGrowthOutside0X128", 0)),
            fee_growth_outside_1_x128=int(data.get("feeGrowthOutside1X128", 0)),
        )


@dataclass
class Position:
    """Position-Indexed State (Section 6.4, Table 3)

    - liquidity: 포지션의 유동성 (l)
    - tickLower: 하한 틱 (i_l)
    - tickUpper: 상한 틱 (i_u)
    - feeGrowthInside0LastX128: 마지막 업데이트 시점의 범위 내 수수료 token0 (f_r,0(t_0))
    - feeGrowthInside1LastX128: 마지막 업데이트 시점의 범위 내 수수료 token1 (f_r,1(t_0))
    - tokensOwed0: 미수령 token0 수수료
    - tokensOwed1: 미수령 token1 수수료
    """
    pool_id: str
    tick_lower: int  # i_l
    tick_upper: int  # i_u
    liquidity: int  # l
    fee_growth_inside_0_last_x128: int = 0  # f_r,0(t_0)
    fee_growth_inside_1_last_x128: int = 0  # f_r,1(t_0)
    tokens_owed_0: int = 0
    tokens_owed_1: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            pool_id=data.get("pool", {}).get("id", "") if isinstance(data.get("pool"), dict) else data.get("pool", ""),
            tick_lower=int(data["tickLower"]),
            tick_upper=int(data["tickUpper"]),
            liquidity=int(data["liquidity"]),
            fee_growth_inside_0_last_x128=int(data.get("feeGrowthInside0LastX128", 0)),
            fee_growth_inside_1_last_x128=int(data.get("feeGrowthInside1LastX128", 0)),
            tokens_owed_0=int(data.get("tokensOwed0", 0)),
            tokens_owed_1=int(data.get("tokensOwed1", 0)),
        )


@dataclass
class PoolHourData:
    """시간별 Pool 데이터 (백테스트용)

    각 시간대의 Pool 상태 스냅샷
    """
    period_start_unix: int  # Unix timestamp
    tick: int  # 해당 시간의 종가 틱
    sqrt_price: int  # sqrtPriceX96
    liquidity: int  # 총 유동성
    high: float  # 기간 내 최고가
    low: float  # 기간 내 최저가
    close: float  # 종가
    fee_growth_global_0_x128: int  # f_g,0
    fee_growth_global_1_x128: int  # f_g,1
    token0_decimals: int = 18
    token1_decimals: int = 18

    @classmethod
    def from_dict(cls, data: dict) -> "PoolHourData":
        pool_data = data.get("pool", {})
        return cls(
            period_start_unix=int(data["periodStartUnix"]),
            tick=int(data.get("tick", 0)),
            sqrt_price=int(data.get("sqrtPrice", 0)),
            liquidity=int(data.get("liquidity", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("close", 0)),
            fee_growth_global_0_x128=int(data.get("feeGrowthGlobal0X128", 0)),
            fee_growth_global_1_x128=int(data.get("feeGrowthGlobal1X128", 0)),
            token0_decimals=int(pool_data.get("token0", {}).get("decimals", 18)),
            token1_decimals=int(pool_data.get("token1", {}).get("decimals", 18)),
        )


@dataclass
class TickWithPool:
    """Pool 정보와 함께 반환되는 Tick 데이터"""
    tick: Tick
    pool_tick: int  # 현재 Pool의 tick
    token0_decimals: int
    token1_decimals: int

    @classmethod
    def from_dict(cls, data: dict) -> "TickWithPool":
        pool_data = data.get("pool", {})
        return cls(
            tick=Tick.from_dict(data),
            pool_tick=int(pool_data.get("tick", 0)),
            token0_decimals=int(pool_data.get("token0", {}).get("decimals", 18)),
            token1_decimals=int(pool_data.get("token1", {}).get("decimals", 18)),
        )
