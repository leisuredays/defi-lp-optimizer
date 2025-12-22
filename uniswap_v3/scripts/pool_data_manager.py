#!/usr/bin/env python3
"""
Pool Data Manager - 풀 데이터 수집 및 관리

The Graph API에서 데이터를 가져오고 SQLite DB로 관리합니다.
중복 방지 및 데이터 무결성을 보장합니다.

Usage:
    # 데이터 수집 (DB에 저장)
    python pool_data_manager.py fetch --pool WETH/USDT --days 365

    # DB 상태 확인
    python pool_data_manager.py status

    # CSV로 내보내기
    python pool_data_manager.py export --pool WETH/USDT --output data.csv

    # 특정 기간 데이터 조회
    python pool_data_manager.py query --pool WETH/USDT --from 2024-01-01 --to 2024-12-31
"""

import sys
import os
import sqlite3
import argparse
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from uniswap_v3.data.graph_client import GraphClient, GraphClientError
from uniswap_v3.constants import CHAIN_IDS


# ============================================================================
# Pool Registry
# ============================================================================

POOL_REGISTRY = {
    "ethereum": {
        "WETH/USDC": {
            "id": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
            "token0": {"symbol": "USDC", "decimals": 6},
            "token1": {"symbol": "WETH", "decimals": 18},
            "fee_tier": 3000,
            "base_token": "WETH",
        },
        "WETH/USDC-0.05": {
            "id": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "token0": {"symbol": "USDC", "decimals": 6},
            "token1": {"symbol": "WETH", "decimals": 18},
            "fee_tier": 500,
            "base_token": "WETH",
        },
        "WETH/USDT": {
            "id": "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
            "token0": {"symbol": "WETH", "decimals": 18},
            "token1": {"symbol": "USDT", "decimals": 6},
            "fee_tier": 3000,
            "base_token": "WETH",
        },
        "WBTC/WETH": {
            "id": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
            "token0": {"symbol": "WBTC", "decimals": 8},
            "token1": {"symbol": "WETH", "decimals": 18},
            "fee_tier": 3000,
            "base_token": "WBTC",
        },
        "WBTC/USDC": {
            "id": "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35",
            "token0": {"symbol": "WBTC", "decimals": 8},
            "token1": {"symbol": "USDC", "decimals": 6},
            "fee_tier": 3000,
            "base_token": "WBTC",
        },
    },
    "polygon": {
        "WETH/USDC": {
            "id": "0x45dda9cb7c25131df268515131f647d726f50608",
            "token0": {"symbol": "WETH", "decimals": 18},
            "token1": {"symbol": "USDC", "decimals": 6},
            "fee_tier": 500,
            "base_token": "WETH",
        },
    },
    "arbitrum": {
        "WETH/USDC": {
            "id": "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",
            "token0": {"symbol": "WETH", "decimals": 18},
            "token1": {"symbol": "USDC", "decimals": 6},
            "fee_tier": 500,
            "base_token": "WETH",
        },
    },
    "optimism": {
        "WETH/USDC": {
            "id": "0x85149247691df622eaf1a8bd0cafd40bc45154a9",
            "token0": {"symbol": "WETH", "decimals": 18},
            "token1": {"symbol": "USDC", "decimals": 6},
            "fee_tier": 500,
            "base_token": "WETH",
        },
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PoolConfig:
    """풀 설정"""
    name: str
    chain: str
    pool_id: str
    token0_symbol: str
    token0_decimals: int
    token1_symbol: str
    token1_decimals: int
    fee_tier: int
    base_token: str

    @property
    def invert_price(self) -> bool:
        """가격을 반전해야 하는지 (base_token이 token1인 경우)"""
        return self.base_token == self.token1_symbol


def get_pool_config(pool_name: str, chain: str = "ethereum") -> Optional[PoolConfig]:
    """풀 설정 가져오기"""
    chain_pools = POOL_REGISTRY.get(chain, {})
    pool_data = chain_pools.get(pool_name)

    if not pool_data:
        return None

    return PoolConfig(
        name=pool_name,
        chain=chain,
        pool_id=pool_data["id"],
        token0_symbol=pool_data["token0"]["symbol"],
        token0_decimals=pool_data["token0"]["decimals"],
        token1_symbol=pool_data["token1"]["symbol"],
        token1_decimals=pool_data["token1"]["decimals"],
        fee_tier=pool_data["fee_tier"],
        base_token=pool_data["base_token"],
    )


# ============================================================================
# Pool Data Database
# ============================================================================

class PoolDataDB:
    """풀 데이터 SQLite 데이터베이스 관리"""

    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'pool_data.db'
    )

    def __init__(self, db_path: str = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH

        # 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._init_db()

    def _init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 풀 메타데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_id TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    name TEXT NOT NULL,
                    token0_symbol TEXT,
                    token0_decimals INTEGER,
                    token1_symbol TEXT,
                    token1_decimals INTEGER,
                    fee_tier INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pool_id, chain)
                )
            ''')

            # 시간별 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pool_hour_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_id TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    period_start_unix INTEGER NOT NULL,
                    tick INTEGER,
                    liquidity TEXT,
                    sqrt_price TEXT,
                    high REAL,
                    low REAL,
                    close REAL,
                    fee_growth_global_0_x128 TEXT,
                    fee_growth_global_1_x128 TEXT,
                    token0_decimals INTEGER,
                    token1_decimals INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pool_id, chain, period_start_unix)
                )
            ''')

            # 인덱스 생성
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pool_hour_data_lookup
                ON pool_hour_data(pool_id, chain, period_start_unix)
            ''')

            conn.commit()

    def save_pool_metadata(self, pool_config: PoolConfig) -> int:
        """풀 메타데이터 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR IGNORE INTO pools
                (pool_id, chain, name, token0_symbol, token0_decimals,
                 token1_symbol, token1_decimals, fee_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pool_config.pool_id,
                pool_config.chain,
                pool_config.name,
                pool_config.token0_symbol,
                pool_config.token0_decimals,
                pool_config.token1_symbol,
                pool_config.token1_decimals,
                pool_config.fee_tier,
            ))

            conn.commit()
            return cursor.lastrowid

    def save_hour_data(self, pool_config: PoolConfig, hour_datas: List[Any]) -> Tuple[int, int]:
        """
        시간별 데이터 저장 (중복 무시)

        Returns:
            (inserted_count, skipped_count)
        """
        if not hour_datas:
            return 0, 0

        inserted = 0
        skipped = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for hd in hour_datas:
                try:
                    cursor.execute('''
                        INSERT INTO pool_hour_data
                        (pool_id, chain, period_start_unix, tick, liquidity, sqrt_price,
                         high, low, close, fee_growth_global_0_x128, fee_growth_global_1_x128,
                         token0_decimals, token1_decimals)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pool_config.pool_id,
                        pool_config.chain,
                        hd.period_start_unix,
                        hd.tick,
                        str(hd.liquidity),
                        str(hd.sqrt_price),
                        hd.high,
                        hd.low,
                        hd.close,
                        str(hd.fee_growth_global_0_x128),
                        str(hd.fee_growth_global_1_x128),
                        hd.token0_decimals,
                        hd.token1_decimals,
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    # 중복 데이터 - UNIQUE 제약 조건 위반
                    skipped += 1

            conn.commit()

        return inserted, skipped

    def get_hour_data(
        self,
        pool_id: str,
        chain: str,
        from_timestamp: int,
        to_timestamp: int
    ) -> List[Dict]:
        """시간별 데이터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
                AND period_start_unix >= ? AND period_start_unix <= ?
                ORDER BY period_start_unix ASC
            ''', (pool_id, chain, from_timestamp, to_timestamp))

            return [dict(row) for row in cursor.fetchall()]

    def get_data_range(self, pool_id: str, chain: str) -> Optional[Tuple[int, int, int]]:
        """
        저장된 데이터 범위 조회

        Returns:
            (min_timestamp, max_timestamp, total_count) or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    MIN(period_start_unix) as min_ts,
                    MAX(period_start_unix) as max_ts,
                    COUNT(*) as total
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
            ''', (pool_id, chain))

            row = cursor.fetchone()
            if row and row[0]:
                return row[0], row[1], row[2]
            return None

    def get_all_pools(self) -> List[Dict]:
        """저장된 모든 풀 목록"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    p.*,
                    MIN(d.period_start_unix) as data_from,
                    MAX(d.period_start_unix) as data_to,
                    COUNT(d.id) as data_count
                FROM pools p
                LEFT JOIN pool_hour_data d ON p.pool_id = d.pool_id AND p.chain = d.chain
                GROUP BY p.id
            ''')

            return [dict(row) for row in cursor.fetchall()]

    def print_status(self):
        """DB 상태 출력"""
        print("\n" + "=" * 70)
        print("Pool Data Database Status")
        print("=" * 70)
        print(f"Database: {self.db_path}")

        pools = self.get_all_pools()

        if not pools:
            print("\n데이터 없음")
            return

        print(f"\n총 {len(pools)}개 풀:\n")
        print(f"{'Pool':<20} {'Chain':<12} {'From':<12} {'To':<12} {'Records':>10}")
        print("-" * 70)

        for p in pools:
            from_date = datetime.fromtimestamp(p['data_from']).strftime('%Y-%m-%d') if p['data_from'] else 'N/A'
            to_date = datetime.fromtimestamp(p['data_to']).strftime('%Y-%m-%d') if p['data_to'] else 'N/A'
            print(f"{p['name']:<20} {p['chain']:<12} {from_date:<12} {to_date:<12} {p['data_count']:>10,}")

        print("=" * 70)

    def export_to_csv(self, pool_id: str, chain: str, output_path: str,
                      from_timestamp: int = None, to_timestamp: int = None):
        """CSV로 내보내기"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = '''
                SELECT * FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
            '''
            params = [pool_id, chain]

            if from_timestamp:
                query += ' AND period_start_unix >= ?'
                params.append(from_timestamp)

            if to_timestamp:
                query += ' AND period_start_unix <= ?'
                params.append(to_timestamp)

            query += ' ORDER BY period_start_unix ASC'

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                print("내보낼 데이터 없음")
                return

            fieldnames = rows[0].keys()

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            print(f"✅ CSV 저장: {output_path} ({len(rows)}개 레코드)")

    def check_integrity(self, pool_id: str, chain: str) -> Dict[str, Any]:
        """
        데이터 무결성 검사

        Returns:
            검사 결과 딕셔너리
        """
        result = {
            'pool_id': pool_id,
            'chain': chain,
            'total_records': 0,
            'date_range': None,
            'expected_hours': 0,
            'missing_hours': 0,
            'missing_pct': 0.0,
            'duplicates': 0,
            'null_ticks': 0,
            'null_fee_growth': 0,
            'tick_range': None,
            'gaps': [],  # 24시간 이상 갭 목록
            'issues': [],
            'is_valid': True,
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. 총 레코드 수 및 날짜 범위
            cursor.execute('''
                SELECT COUNT(*), MIN(period_start_unix), MAX(period_start_unix)
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
            ''', (pool_id, chain))

            row = cursor.fetchone()
            if not row or not row[1]:
                result['issues'].append("데이터 없음")
                result['is_valid'] = False
                return result

            result['total_records'] = row[0]
            min_ts, max_ts = row[1], row[2]
            result['date_range'] = {
                'from': datetime.fromtimestamp(min_ts).isoformat(),
                'to': datetime.fromtimestamp(max_ts).isoformat(),
            }

            # 2. 예상 시간 수 계산 (1시간 = 3600초)
            expected_hours = ((max_ts - min_ts) // 3600) + 1
            result['expected_hours'] = expected_hours
            result['missing_hours'] = expected_hours - result['total_records']
            result['missing_pct'] = (result['missing_hours'] / expected_hours) * 100 if expected_hours > 0 else 0

            # 3. 중복 검사
            cursor.execute('''
                SELECT period_start_unix, COUNT(*) as cnt
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
                GROUP BY period_start_unix
                HAVING cnt > 1
            ''', (pool_id, chain))

            duplicates = cursor.fetchall()
            result['duplicates'] = len(duplicates)
            if duplicates:
                result['issues'].append(f"중복 타임스탬프 {len(duplicates)}개 발견")
                result['is_valid'] = False

            # 4. NULL 값 검사
            cursor.execute('''
                SELECT
                    SUM(CASE WHEN tick IS NULL THEN 1 ELSE 0 END) as null_ticks,
                    SUM(CASE WHEN fee_growth_global_0_x128 IS NULL OR fee_growth_global_0_x128 = '' THEN 1 ELSE 0 END) as null_fg
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
            ''', (pool_id, chain))

            null_row = cursor.fetchone()
            result['null_ticks'] = null_row[0] or 0
            result['null_fee_growth'] = null_row[1] or 0

            if result['null_ticks'] > 0:
                result['issues'].append(f"NULL tick 값 {result['null_ticks']}개")

            # 5. tick 범위 검사
            cursor.execute('''
                SELECT MIN(tick), MAX(tick)
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ? AND tick IS NOT NULL
            ''', (pool_id, chain))

            tick_row = cursor.fetchone()
            if tick_row[0] is not None:
                result['tick_range'] = {'min': tick_row[0], 'max': tick_row[1]}

            # 6. 큰 갭 찾기 (24시간 이상)
            cursor.execute('''
                SELECT period_start_unix
                FROM pool_hour_data
                WHERE pool_id = ? AND chain = ?
                ORDER BY period_start_unix ASC
            ''', (pool_id, chain))

            timestamps = [row[0] for row in cursor.fetchall()]
            gaps = []
            for i in range(1, len(timestamps)):
                gap_hours = (timestamps[i] - timestamps[i-1]) / 3600
                if gap_hours > 24:
                    gaps.append({
                        'from': datetime.fromtimestamp(timestamps[i-1]).isoformat(),
                        'to': datetime.fromtimestamp(timestamps[i]).isoformat(),
                        'hours': int(gap_hours),
                    })

            result['gaps'] = gaps[:10]  # 최대 10개만
            if len(gaps) > 10:
                result['issues'].append(f"24시간+ 갭 {len(gaps)}개 발견 (상위 10개만 표시)")

            # 7. 최종 유효성 판단
            if result['missing_pct'] > 20:
                result['issues'].append(f"누락율 {result['missing_pct']:.1f}% (20% 초과)")
                result['is_valid'] = False

        return result

    def print_integrity_report(self, pool_id: str, chain: str):
        """무결성 검사 보고서 출력"""
        result = self.check_integrity(pool_id, chain)

        print("\n" + "=" * 70)
        print("데이터 무결성 검사 보고서")
        print("=" * 70)
        print(f"Pool ID: {result['pool_id']}")
        print(f"Chain: {result['chain']}")

        if not result['date_range']:
            print("\n❌ 데이터 없음")
            return result

        print(f"\n📅 데이터 범위:")
        print(f"   {result['date_range']['from']} ~ {result['date_range']['to']}")

        print(f"\n📊 레코드 통계:")
        print(f"   총 레코드: {result['total_records']:,}개")
        print(f"   예상 시간: {result['expected_hours']:,}시간")
        print(f"   누락 시간: {result['missing_hours']:,}시간 ({result['missing_pct']:.2f}%)")

        print(f"\n🔍 무결성 검사:")
        print(f"   중복: {result['duplicates']}개")
        print(f"   NULL tick: {result['null_ticks']}개")
        print(f"   NULL fee_growth: {result['null_fee_growth']}개")

        if result['tick_range']:
            print(f"\n📈 Tick 범위:")
            print(f"   Min: {result['tick_range']['min']}")
            print(f"   Max: {result['tick_range']['max']}")

        if result['gaps']:
            print(f"\n⚠️  큰 갭 (24시간+):")
            for gap in result['gaps']:
                print(f"   {gap['from']} ~ {gap['to']} ({gap['hours']}시간)")

        print("\n" + "-" * 70)
        if result['is_valid']:
            print("✅ 무결성 검사 통과")
        else:
            print("❌ 무결성 문제 발견:")
            for issue in result['issues']:
                print(f"   - {issue}")

        print("=" * 70)

        return result


# ============================================================================
# Data Fetcher
# ============================================================================

class PoolDataFetcher:
    """풀 데이터 수집기"""

    def __init__(self, db: PoolDataDB = None, verbose: bool = True):
        self.db = db or PoolDataDB()
        self.verbose = verbose
        self._clients: Dict[str, GraphClient] = {}

    def _get_client(self, chain: str) -> GraphClient:
        """체인별 클라이언트 (캐싱)"""
        if chain not in self._clients:
            self._clients[chain] = GraphClient(chain=chain)
        return self._clients[chain]

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def fetch_and_save(
        self,
        pool_config: PoolConfig,
        days: int = 365,
        end_date: datetime = None
    ) -> Tuple[int, int]:
        """
        데이터 수집 및 DB 저장

        Returns:
            (inserted_count, skipped_count)
        """
        end_dt = end_date or datetime.now()
        start_dt = end_dt - timedelta(days=days)

        from_ts = int(start_dt.timestamp())
        to_ts = int(end_dt.timestamp())

        self._log(f"\n{'='*60}")
        self._log(f"풀: {pool_config.name} ({pool_config.chain})")
        self._log(f"기간: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days}일)")
        self._log(f"{'='*60}")

        # 기존 데이터 범위 확인
        existing_range = self.db.get_data_range(pool_config.pool_id, pool_config.chain)
        if existing_range:
            existing_from = datetime.fromtimestamp(existing_range[0])
            existing_to = datetime.fromtimestamp(existing_range[1])
            self._log(f"기존 데이터: {existing_from.strftime('%Y-%m-%d')} ~ {existing_to.strftime('%Y-%m-%d')} ({existing_range[2]:,}개)")

        # 데이터 가져오기
        try:
            client = self._get_client(pool_config.chain)
            self._log(f"\nThe Graph API에서 데이터 가져오는 중...")

            hour_datas = client.get_pool_hour_datas_paginated(
                pool_config.pool_id, from_ts, to_ts
            )
            hour_datas.sort(key=lambda x: x.period_start_unix)

            self._log(f"가져온 데이터: {len(hour_datas)}개")

        except GraphClientError as e:
            self._log(f"❌ API 오류: {e}")
            return 0, 0

        if not hour_datas:
            self._log("데이터 없음")
            return 0, 0

        # 메타데이터 저장
        self.db.save_pool_metadata(pool_config)

        # 시간별 데이터 저장
        inserted, skipped = self.db.save_hour_data(pool_config, hour_datas)

        self._log(f"\n✅ 저장 완료:")
        self._log(f"   - 새로 추가: {inserted:,}개")
        self._log(f"   - 중복 스킵: {skipped:,}개")

        return inserted, skipped

    def fetch_multiple(
        self,
        pools: List[Tuple[str, str]],  # [(pool_name, chain), ...]
        days: int = 365,
        end_date: datetime = None
    ) -> Dict[str, Tuple[int, int]]:
        """여러 풀 데이터 수집"""
        results = {}

        for pool_name, chain in pools:
            pool_config = get_pool_config(pool_name, chain)
            if not pool_config:
                self._log(f"⚠️  풀 없음: {pool_name} ({chain})")
                continue

            inserted, skipped = self.fetch_and_save(pool_config, days, end_date)
            results[f"{pool_name}_{chain}"] = (inserted, skipped)

        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pool Data Manager - 풀 데이터 수집 및 관리',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 데이터 수집
  python pool_data_manager.py fetch --pool WETH/USDT --days 365

  # 여러 풀 수집
  python pool_data_manager.py fetch --pool WETH/USDT,WETH/USDC --days 365

  # DB 상태 확인
  python pool_data_manager.py status

  # 무결성 검사
  python pool_data_manager.py check --pool WETH/USDT

  # CSV 내보내기
  python pool_data_manager.py export --pool WETH/USDT --output data.csv

  # 사용 가능한 풀 목록
  python pool_data_manager.py list-pools
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='명령어')

    # fetch 명령
    fetch_parser = subparsers.add_parser('fetch', help='데이터 수집')
    fetch_parser.add_argument('--pool', type=str, required=True,
                              help='풀 이름 (쉼표로 구분). 예: WETH/USDT,WETH/USDC')
    fetch_parser.add_argument('--chain', type=str, default='ethereum',
                              choices=list(CHAIN_IDS.keys()),
                              help='체인 (default: ethereum)')
    fetch_parser.add_argument('--days', type=int, default=365,
                              help='수집 기간 (일, default: 365)')

    # status 명령
    subparsers.add_parser('status', help='DB 상태 확인')

    # export 명령
    export_parser = subparsers.add_parser('export', help='CSV 내보내기')
    export_parser.add_argument('--pool', type=str, required=True,
                               help='풀 이름')
    export_parser.add_argument('--chain', type=str, default='ethereum',
                               help='체인')
    export_parser.add_argument('--output', type=str, required=True,
                               help='출력 파일 경로')
    export_parser.add_argument('--from', dest='from_date', type=str,
                               help='시작일 (YYYY-MM-DD)')
    export_parser.add_argument('--to', dest='to_date', type=str,
                               help='종료일 (YYYY-MM-DD)')

    # list-pools 명령
    subparsers.add_parser('list-pools', help='사용 가능한 풀 목록')

    # check 명령 (무결성 검사)
    check_parser = subparsers.add_parser('check', help='데이터 무결성 검사')
    check_parser.add_argument('--pool', type=str, required=True,
                              help='풀 이름')
    check_parser.add_argument('--chain', type=str, default='ethereum',
                              help='체인')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = PoolDataDB()

    if args.command == 'status':
        db.print_status()

    elif args.command == 'list-pools':
        print("\n사용 가능한 풀 목록:")
        print("=" * 50)
        for chain, pools in POOL_REGISTRY.items():
            print(f"\n{chain.upper()}:")
            for pool_name in pools:
                print(f"  - {pool_name}")
        print()

    elif args.command == 'fetch':
        pool_names = [p.strip() for p in args.pool.split(',')]
        pools = [(name, args.chain) for name in pool_names]

        fetcher = PoolDataFetcher(db=db)
        results = fetcher.fetch_multiple(pools, days=args.days)

        print("\n" + "=" * 50)
        print("수집 결과 요약:")
        for key, (inserted, skipped) in results.items():
            print(f"  {key}: +{inserted:,} 새로 추가, {skipped:,} 중복 스킵")
        print("=" * 50)

    elif args.command == 'export':
        pool_config = get_pool_config(args.pool, args.chain)
        if not pool_config:
            print(f"❌ 풀 없음: {args.pool} ({args.chain})")
            return

        from_ts = None
        to_ts = None

        if args.from_date:
            from_ts = int(datetime.strptime(args.from_date, '%Y-%m-%d').timestamp())
        if args.to_date:
            to_ts = int(datetime.strptime(args.to_date, '%Y-%m-%d').timestamp())

        db.export_to_csv(
            pool_config.pool_id,
            pool_config.chain,
            args.output,
            from_ts,
            to_ts
        )

    elif args.command == 'check':
        pool_config = get_pool_config(args.pool, args.chain)
        if not pool_config:
            print(f"❌ 풀 없음: {args.pool} ({args.chain})")
            return

        db.print_integrity_report(pool_config.pool_id, pool_config.chain)


if __name__ == "__main__":
    main()
