# uniswap_v3 Python Module

Uniswap V3 LP 전략을 위한 강화학습 훈련 및 백테스트 모듈.

## 디렉토리 구조

```
uniswap_v3/
├── constants.py          # Q96, Q128, TICK_SPACINGS 등 상수
├── math/                 # Uniswap V3 수학 공식
│   ├── calc.py           # L 계산, fee 계산, 토큰 분할
│   └── convert.py        # tick ↔ price ↔ sqrt 변환
├── ml/                   # 강화학습 모듈
│   ├── environment.py    # Gymnasium 환경 (UniswapV3LPEnv)
│   ├── adapter.py        # IL, LVR 계산 어댑터
│   ├── trainer.py        # PPO 훈련 래퍼
│   ├── callbacks.py      # 훈련 콜백
│   └── regime.py         # 시장 레짐 감지 (HMM)
├── data/                 # 데이터 수집 및 저장
│   ├── graph_client.py   # The Graph API 클라이언트
│   ├── queries.py        # GraphQL 쿼리
│   ├── types.py          # 데이터 타입 정의
│   └── pool_data.db      # SQLite DB (시간별 풀 데이터)
├── scripts/              # 실행 스크립트
├── tests/                # pytest 테스트 (테스트 파일 저장 위치)
├── models/               # 저장된 모델 (.zip)
└── docs/                 # 참조 문서, 논문
```

## 핵심 모듈

### math/
Uniswap V3 Whitepaper 기반 수학 공식:
- `amounts_to_L()`: 토큰 수량 → 유동성 L
- `L_to_amounts()`: 유동성 L → 토큰 수량
- `tick_to_price()`, `price_to_tick()`: tick ↔ 가격 변환
- `fee_delta()`: fee_growth overflow 처리

### ml/environment.py
`UniswapV3LPEnv`: Gymnasium 환경
- **Action Space**: `[-1, 1]^3` (rebalance, lower_delta, upper_delta)
- **Observation**: 28차원 (가격, 변동성, 포지션 상태 등)
- **Reward**: `fees - LVR - gas`

### data/graph_client.py
The Graph 분산 네트워크에서 풀 데이터 수집.
- 시간별 데이터: `poolHourDatas`
- fee_growth_global 포함

## 스크립트 사용법

### 데이터 수집
```bash
cd uniswap_v3/scripts
python pool_data_manager.py fetch --pool WETH/USDT --days 365
python pool_data_manager.py status
```

### 훈련
```bash
# Walk-Forward 전체 훈련
python train_wfe.py

# 특정 Fold만 훈련
python train_fold.py --fold 1 --total 2000000

# 훈련 재개
python train_continue.py --fold 1 --checkpoint fold_01_500k.zip
```

### 평가
```bash
# 베이스라인 비교
python evaluate_baselines.py --model fold_01.zip --fold 1

# 모델 평가
python evaluate_model.py --model fold_01.zip --fold 1
```

### 백테스트
```bash
python backtest.py --pool WETH/USDT --range 10,20,30
python backtest.py --pool WETH/USDT --range 20 --from 2024-01-01 --to 2024-12-31
```

### 시각화
```bash
python visualize_fold.py --fold 1
python visualize_model.py --model fold_01.zip --start 0 --length 720
```

## 환경 설정

**conda 환경 필수**:
```bash
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend
```

## 주요 상수 (constants.py)

```python
Q96 = 2**96      # sqrt price 스케일
Q128 = 2**128    # fee_growth 스케일
TICK_SPACINGS = {500: 10, 3000: 60, 10000: 200}
```

## 테스트

테스트 파일은 `tests/` 디렉토리에 저장:

```
tests/
├── __init__.py
├── test_math.py         # math 모듈 테스트
├── test_environment.py  # ML 환경 테스트
├── test_backtest.py     # 백테스트 로직 테스트
└── test_*.py            # 기타 테스트
```

**네이밍 규칙**: `test_<모듈명>.py`

```bash
# 전체 테스트
cd uniswap_v3
pytest tests/ -v

# 특정 테스트
pytest tests/test_math.py -v
```
