"""
Uniswap V3 상수 정의

온체인 수준 정밀도를 위한 상수들:
- Q96: sqrt price 인코딩에 사용 (2^96)
- Q128: fee growth 인코딩에 사용 (2^128)
- FEE_TIERS: 지원되는 수수료 티어
- TICK_SPACINGS: 각 수수료 티어별 틱 간격
"""

from typing import Dict

# Fixed-point 인코딩 상수
Q96: int = 2 ** 96
Q128: int = 2 ** 128
Q192: int = 2 ** 192

# 수수료 티어 (basis points)
# 500 = 0.05%, 3000 = 0.30%, 10000 = 1.00%
FEE_TIERS: Dict[int, str] = {
    100: "0.01%",    # 1 bps
    500: "0.05%",    # 5 bps
    3000: "0.30%",   # 30 bps
    10000: "1.00%",  # 100 bps
}

# 각 수수료 티어별 틱 간격
# tick_spacing = fee_tier / 50 (대략적으로)
TICK_SPACINGS: Dict[int, int] = {
    100: 1,
    500: 10,
    3000: 60,
    10000: 200,
}

# 지원되는 체인 ID 및 Subgraph ID
CHAIN_IDS: Dict[str, int] = {
    "ethereum": 0,
    "optimism": 1,
    "arbitrum": 2,
    "polygon": 3,
    "celo": 5,
}

# The Graph Subgraph IDs
SUBGRAPH_IDS: Dict[int, str] = {
    0: "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",  # Ethereum Mainnet
    1: "Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj",  # Optimism
    2: "FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM",  # Arbitrum
    3: "3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm",  # Polygon
    5: "ESdrTJ3twMwWVoQ1hUE2u7PugEHX3QkenudD6aXCkDQ4",  # Celo
}

# 틱 범위 상수
MIN_TICK: int = -887272
MAX_TICK: int = 887272

# sqrt(1.0001) 상수 (tick 계산에 사용)
SQRT_RATIO_BASE: float = 1.0001

# uint256 최대값
UINT256_MAX: int = 2 ** 256 - 1
UINT128_MAX: int = 2 ** 128 - 1
