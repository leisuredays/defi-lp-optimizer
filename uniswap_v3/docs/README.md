# Uniswap V3 백테스트 도구

Uniswap V3 유동성 포지션의 과거 성과를 분석하는 Python 도구입니다.

## 구조

```
uniswap_v3/
├── data/               # 데이터베이스 및 API 클라이언트
│   ├── pool_data.db    # 풀 시간별 데이터 (SQLite)
│   ├── graph_client.py # The Graph API 클라이언트
│   ├── queries.py      # GraphQL 쿼리
│   └── types.py        # 데이터 타입 정의
├── math/               # Uniswap V3 수학 함수
├── core/               # 핵심 로직
├── scripts/            # CLI 도구
│   ├── pool_data_manager.py  # 데이터 수집/관리
│   ├── backtest.py           # 백테스트 실행
│   └── multi_pool_analyzer_v2.py  # 다중 풀 비교
└── docs/               # 문서
```

## 환경 설정

```bash
# conda 환경 활성화
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend
```

## 도구 사용법

### 1. 데이터 수집 (pool_data_manager.py)

The Graph API에서 풀 시간별 데이터를 수집하여 SQLite DB에 저장합니다.

```bash
cd /home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/scripts

# 데이터 수집 (기본: WETH/USDT, 365일)
python pool_data_manager.py fetch

# 특정 풀과 기간 지정
python pool_data_manager.py fetch --pool "WETH/USDC" --days 180

# 특정 체인 지정
python pool_data_manager.py fetch --pool "WETH/USDC" --chain polygon --days 90

# DB 상태 확인
python pool_data_manager.py status

# 특정 풀 상태 확인
python pool_data_manager.py status --pool "WETH/USDT"

# 저장된 풀 목록
python pool_data_manager.py list-pools

# 데이터 무결성 검사
python pool_data_manager.py check

# CSV로 내보내기
python pool_data_manager.py export --pool "WETH/USDT" --output data.csv
```

**지원 체인:** ethereum (기본), polygon, arbitrum, optimism, celo

### 2. 백테스트 (backtest.py)

수집된 데이터로 유동성 포지션 성과를 분석합니다.

```bash
# 기본 백테스트 (DB에서 데이터 로드)
python backtest.py

# 풀 지정
python backtest.py --pool "WETH/USDT"

# 투자금액 및 범위 지정
python backtest.py --pool "WETH/USDT" --investment 10000 --range 20

# 기간 지정
python backtest.py --pool "WETH/USDT" --start 2024-01-01 --end 2024-12-31

# 차트 표시 (창으로)
python backtest.py --pool "WETH/USDT" --range 20 --chart

# 차트 저장
python backtest.py --pool "WETH/USDT" --range 20 --chart output.png

# 가격 반전 (WETH/USDC 등 token0이 스테이블코인인 경우)
python backtest.py --pool "WETH/USDC" --range 20 --chart --invert

# 여러 범위 비교
python backtest.py --pool "WETH/USDT" --range 10,20,30 --output results.csv

# CSV 파일에서 직접 로드
python backtest.py --csv data.csv --range 20
```

**주요 옵션:**
| 옵션 | 설명 |
|------|------|
| `--pool` | 풀 이름 (예: "WETH/USDT") |
| `--chain` | 체인 (ethereum, polygon, ...) |
| `--range` | 가격 범위 % (예: 20 = ±20%) |
| `--investment` | 초기 투자금 (USD) |
| `--start`, `--end` | 분석 기간 |
| `--chart` | 차트 표시 (경로 지정시 파일로 저장) |
| `--invert` | 가격 반전 (token0이 quote인 경우) |
| `--output` | 결과 CSV 저장 경로 |

### 3. 다중 풀 비교 (multi_pool_analyzer_v2.py)

여러 풀과 조건을 동시에 비교 분석합니다.

```bash
# 여러 풀 비교
python multi_pool_analyzer_v2.py --pools WETH/USDC,WETH/USDT,WBTC/WETH

# 여러 기간 비교
python multi_pool_analyzer_v2.py --periods 7,30,90

# 여러 범위 비교
python multi_pool_analyzer_v2.py --ranges 10,20,50

# 전체 옵션
python multi_pool_analyzer_v2.py \
    --pools WETH/USDC,WETH/USDT \
    --periods 7,30,90 \
    --ranges 10,20,50 \
    --investment 10000 \
    --output results \
    --format csv,json,chart
```

## 출력 지표

| 지표 | 설명 |
|------|------|
| `total_fees_usd` | 총 수수료 수익 (USD) |
| `final_lp_value` | 최종 LP 포지션 가치 |
| `final_hodl_value` | HODL 전략 가치 |
| `il_usd`, `il_pct` | 비영구 손실 (USD, %) |
| `net_pnl_usd`, `net_pnl_pct` | 순수익 (USD, %) |
| `vs_hodl_usd`, `vs_hodl_pct` | HODL 대비 수익 |
| `apr_pct` | 연간 수익률 |
| `active_pct` | 범위 내 활성 비율 |

## 차트 패널

6개 패널 차트가 표시됩니다:

1. **Price Trend** - 가격 추이 및 LP 범위
2. **Position Value** - 포지션 가치 vs HODL
3. **Cumulative Fees** - 누적 수수료
4. **IL Impact** - 비영구 손실 추이
5. **Net PnL** - 순손익 추이
6. **Range Activity** - 범위 내/외 상태

## 가격 반전 (--invert)

Uniswap 풀의 token0/token1 순서는 컨트랙트 주소에 의해 결정됩니다:

- **WETH/USDT (Ethereum):** token0=WETH, token1=USDT → 반전 불필요
- **WETH/USDC (Ethereum):** token0=USDC, token1=WETH → `--invert` 필요

`--invert` 옵션은 가격을 1/price로 표시합니다 (예: 0.0003 → 3000).

## 데이터베이스

SQLite 데이터베이스 위치: `uniswap_v3/data/pool_data.db`

```sql
-- 테이블 스키마
CREATE TABLE pool_hour_data (
    pool_name TEXT,
    chain TEXT,
    period_start_unix INTEGER,
    tick INTEGER,
    liquidity TEXT,
    sqrt_price TEXT,
    high TEXT,
    low TEXT,
    close TEXT,
    fee_growth_global_0_x128 TEXT,
    fee_growth_global_1_x128 TEXT,
    PRIMARY KEY (pool_name, chain, period_start_unix)
);
```
