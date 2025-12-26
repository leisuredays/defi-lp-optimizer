# 2. Regime Detection (Hidden Markov Model)

### 왜 필요한가?

LP 전략의 핵심 딜레마:
- **좁은 범위**: 높은 수수료 수익 BUT 범위 이탈 시 IL 폭발
- **넓은 범위**: 안전하지만 수수료 수익 낮음

최적 전략은 **시장 레짐에 따라 다릅니다**:

| 레짐 | 특성 | 최적 LP 전략 |
|------|------|-------------|
| 저변동성 횡보 | σ 낮음, 추세 없음 | 좁은 범위 (2σ 이하) |
| 고변동성 추세 | σ 높음, 방향성 있음 | 넓은 범위 (4σ 이상) 또는 대기 |
| 고변동성 횡보 | σ 높음, 추세 없음 | 중간 범위 (3σ) |

### Hidden Markov Model 기초

HMM은 관측되지 않는 "숨겨진 상태"가 관측 데이터를 생성한다고 가정합니다.
```
숨겨진 상태 (레짐):  [Bull] → [Bull] → [Bear] → [Bear] → [Bull]
                      ↓        ↓        ↓        ↓        ↓
관측 데이터 (수익률): +2%     +1%      -3%      -2%      +1%
핵심 파라미터:

전이 확률 (Transition Matrix): P(다음 레짐 | 현재 레짐)
방출 확률 (Emission): 각 레짐에서의 수익률 분포 (보통 Gaussian)

2-State Gaussian HMM 구현
pythonimport numpy as np
from hmmlearn import hmm

class RegimeDetector:
    def __init__(self, n_regimes=2):
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.n_regimes = n_regimes
        
    def fit(self, returns: np.ndarray):
        """
        수익률 시계열로 HMM 학습
        returns: shape (T, n_features) - 예: [returns, volatility]
        """
        # 2D array 필요
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
            
        self.model.fit(returns)
        
        # 레짐 해석 (변동성 기준 정렬)
        means = self.model.means_.flatten()
        covars = np.sqrt(self.model.covars_.flatten())
        
        # 낮은 변동성 = 레짐 0, 높은 변동성 = 레짐 1
        self.regime_order = np.argsort(covars)
        
        print("=== Regime Characteristics ===")
        for i, regime_idx in enumerate(self.regime_order):
            print(f"Regime {i}: mean={means[regime_idx]:.4f}, "
                  f"std={covars[regime_idx]:.4f}")
                  
    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        각 시점의 레짐 확률 반환
        returns: shape (T,) or (T, n_features)
        output: shape (T, n_regimes)
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
            
        # Forward-backward algorithm으로 확률 계산
        posteriors = self.model.predict_proba(returns)
        
        # 레짐 순서 재정렬 (0=저변동성, 1=고변동성)
        return posteriors[:, self.regime_order]
        
    def get_current_regime(self, returns: np.ndarray) -> int:
        """가장 최근 시점의 레짐 반환"""
        proba = self.predict_proba(returns)
        return np.argmax(proba[-1])
관찰 공간에 레짐 피처 추가
pythondef _get_observation(self) -> np.ndarray:
    # 기존 24개 피처
    existing_features = self._get_existing_features()  # shape: (24,)
    
    # 레짐 피처 추가
    recent_returns = self.get_recent_returns(lookback=100)
    regime_proba = self.regime_detector.predict_proba(recent_returns)
    
    # 마지막 시점의 레짐 확률
    regime_features = regime_proba[-1]  # shape: (2,) for 2-state HMM
    
    # 레짐 전환 신호 (급격한 확률 변화)
    if len(regime_proba) > 10:
        regime_momentum = regime_proba[-1] - regime_proba[-10].mean(axis=0)
    else:
        regime_momentum = np.zeros(2)
    
    # 새로운 관찰 공간: 24 + 2 + 2 = 28차원
    return np.concatenate([
        existing_features,      # 24개
        regime_features,        # 2개: P(저변동성), P(고변동성)
        regime_momentum         # 2개: 레짐 전환 속도
    ])