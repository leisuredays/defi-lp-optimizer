
# Backtest Date Range Analysis

## 🔍 Current Limitation

**현재 상태**: Strategy Backtest는 최대 **30일**까지만 조회 가능

**제한 위치**:
- File: `src/layout/StrategyBacktest.js`
- Line 280: `<RangeSlider ... min={5} max={30} ...>`

## 📊 기술적 분석 결과

### The Graph API 테스트

| 요청 기간 | 반환된 데이터 | 실제 커버리지 | 상태 |
|----------|--------------|--------------|------|
| 30일 | 720 hourly records | 30.0일 | ✅ 정상 |
| 41일 | 984 hourly records | 41.0일 | ✅ 정상 |
| 60일 | 1000 hourly records | 41.6일 | ⚠️ API 제한 |
| 90일 | 1000 hourly records | 41.6일 | ⚠️ API 제한 |

### 결론

**30일 제한은 UI/UX 디자인 선택이며, 기술적 제한이 아닙니다.**

#### 기술적 최대값
- **The Graph API 제한**: 쿼리당 최대 **1000개** 결과
- **시간별 데이터**: 1000 hours = **41.6일**
- **실제 권장**: **~41일**까지 안정적으로 확장 가능

#### 현재 코드 상황
```javascript
// src/api/thegraph/uniPoolHourDatas.js:6
// API는 이미 1000개까지 요청하도록 설정됨
first: 1000

// src/layout/StrategyBacktest.js:280
// UI 슬라이더만 30일로 제한
<RangeSlider ... min={5} max={30} ...>
```

## 🚀 확장 방법

### Option 1: 단순 확장 (권장)

슬라이더 최대값을 41일로 변경:

```javascript
// src/layout/StrategyBacktest.js:280
<RangeSlider
  className={styles['range-slider-backtest-days']}
  handleInputChange={handleDaysChange}
  min={5}
  max={41}  // 30 → 41로 변경
  value={days}
  step={1}>
</RangeSlider>
```

### Option 2: 더 긴 기간 (페이지네이션 필요)

60일 이상을 원하는 경우:
1. **페이지네이션** 구현 필요
2. 여러 번의 API 호출로 데이터 조합
3. 더 복잡한 구현

```javascript
// 예시: 90일 데이터를 3번의 쿼리로 가져오기
// Query 1: days 1-30 (1000 records)
// Query 2: days 31-60 (1000 records)
// Query 3: days 61-90 (1000 records)
```

## ⚠️ 고려사항

### 1. 성능
- **30일**: 720 hourly records → 빠른 로딩
- **41일**: 984 hourly records → 약간 느림 (여전히 괜찮음)
- **60일+**: 여러 API 호출 → 현저히 느림

### 2. 사용자 경험
- 더 긴 기간 = 더 많은 데이터 = 더 느린 차트 렌더링
- 모바일에서는 성능 저하 가능

### 3. API 비용
- The Graph는 쿼리 횟수/복잡도로 과금
- 페이지네이션 사용 시 비용 증가

## 💡 권장 사항

### 즉시 적용 가능 (권장)
```bash
# 최대 41일로 확장
max={30} → max={41}
```

**장점**:
- ✅ 코드 1줄만 수정
- ✅ API 호출 변경 없음
- ✅ 성능 영향 최소
- ✅ 37% 더 긴 백테스트 기간

**단점**:
- ⚠️ 41일 이상은 불가능

### 장기적 개선안
더 긴 기간이 필요한 경우:
1. **일별 데이터** 사용 (90일치 이미 가능)
2. **혼합 방식**: 최근 30일은 시간별, 이전 데이터는 일별
3. **캐싱**: 자주 조회되는 풀 데이터 캐싱

## 📝 구현 예시

### 41일로 확장 (1분 작업)

```diff
// src/layout/StrategyBacktest.js

- <RangeSlider className={styles['range-slider-backtest-days']} handleInputChange={handleDaysChange} min={5} max={30} value={days} step={1}></RangeSlider>
+ <RangeSlider className={styles['range-slider-backtest-days']} handleInputChange={handleDaysChange} min={5} max={41} value={days} step={1}></RangeSlider>
```

### 테스트 결과
```
✅ 30일: 720시간 데이터 (30.0일 커버리지)
✅ 41일: 984시간 데이터 (41.0일 커버리지)
⚠️ 60일: 1000시간 데이터 (41.6일 커버리지 - API 제한)
```

## 🎯 결론

**30일 제한은 의도적인 UI 설정이며, 쉽게 41일로 확장 가능합니다.**

- **기술적 제한**: 없음 (API는 이미 1000개 요청)
- **실제 제한**: The Graph API의 1000개 결과 한도
- **최대 가능**: 약 41일
- **권장 변경**: `max={30}` → `max={41}`

변경 시 사용자는 36% 더 긴 백테스트 기간을 얻을 수 있습니다.
