"""
Uniswap V3 Math Adapter - Re-export from uniswap_v3.ml

모든 로직은 uniswap_v3/ml/adapter.py에 위치.
이 파일은 하위 호환성을 위한 re-export.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Re-export all from uniswap_v3.ml.adapter
from uniswap_v3.ml.adapter import *

# 추가 별칭 (하위 호환)
from uniswap_v3.ml.adapter import (
    snap_price as round_to_nearest_tick,
    calc_liquidity as calculate_liquidity,
    calc_il as calculate_il,
    calc_fees as calculate_fees,
    active_ratio as active_liquidity_for_candle,
)

# uniswap_v3.math에서 직접 필요한 상수/함수
from uniswap_v3.constants import Q96, Q128, TICK_SPACINGS, MIN_TICK, MAX_TICK
from uniswap_v3.math import (
    tick_to_sqrt,
    snap_tick,
    amounts_to_L,
    L_to_amounts,
    fee_delta,
    split_tokens,
    # 하위 호환 별칭
    get_sqrt_ratio_at_tick,
    round_tick_to_spacing,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    calculate_fee_growth_delta,
)

# optimal_token_split 별칭
optimal_token_split = split_tokens
