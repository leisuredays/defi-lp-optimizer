"""
Market Regime Detection using Hidden Markov Model (HMM)

This module provides regime detection for LP strategy optimization.
- Low volatility regime: Narrow range optimal (2σ or less)
- High volatility regime: Wide range optimal (4σ or more)

Reference: uniswap_v3/docs/HMM.md
"""

import numpy as np
from hmmlearn import hmm
from typing import Tuple, Optional
import warnings


class RegimeDetector:
    """
    2-State Gaussian HMM for market regime detection.

    States:
        - Regime 0: Low volatility (stable, sideways market)
        - Regime 1: High volatility (trending or volatile market)

    Usage:
        detector = RegimeDetector(n_regimes=2)
        detector.fit(returns)  # Fit on historical returns
        proba = detector.predict_proba(recent_returns)  # Get regime probabilities
    """

    def __init__(self, n_regimes: int = 2, lookback: int = 100):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of hidden states (default 2: low/high volatility)
            lookback: Number of periods for regime detection (default 100 hours)
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.is_fitted = False
        self.regime_order = None  # Maps internal states to [low_vol, high_vol]

        # Suppress convergence warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
                verbose=False
            )

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        """
        Fit HMM on historical returns data.

        Args:
            returns: Array of returns, shape (T,) or (T, n_features)
                     Typically log returns or percentage returns

        Returns:
            self (for chaining)
        """
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Remove NaN/inf values
        valid_mask = np.isfinite(returns).all(axis=1)
        clean_returns = returns[valid_mask]

        if len(clean_returns) < self.lookback:
            raise ValueError(f"Insufficient data: {len(clean_returns)} < {self.lookback}")

        # Fit HMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(clean_returns)

        # Order regimes by volatility (covariance)
        # Lower variance = regime 0 (low vol), higher variance = regime 1 (high vol)
        # covars_ shape depends on covariance_type:
        # - "full": (n_components, n_features, n_features)
        # - "diag": (n_components, n_features)
        # - "spherical": (n_components,)
        if self.model.covars_.ndim == 3:
            # "full" covariance: extract variance (diagonal) for each component
            covars = np.array([np.sqrt(np.diag(self.model.covars_[i])).mean()
                              for i in range(self.n_regimes)])
        elif self.model.covars_.ndim == 2:
            # "diag" covariance
            covars = np.sqrt(self.model.covars_.mean(axis=1))
        else:
            # "spherical" covariance
            covars = np.sqrt(self.model.covars_)
        self.regime_order = np.argsort(covars)

        self.is_fitted = True

        # Print regime characteristics
        means = self.model.means_.flatten()
        print(f"=== HMM Regime Detection Fitted ===")
        for i, regime_idx in enumerate(self.regime_order):
            regime_type = "Low Vol" if i == 0 else "High Vol"
            print(f"  Regime {i} ({regime_type}): mean={means[regime_idx]:.6f}, "
                  f"std={covars[regime_idx]:.6f}")

        return self

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for each time step.

        Args:
            returns: Array of returns, shape (T,) or (T, n_features)

        Returns:
            Array of shape (T, n_regimes) with probabilities.
            Column 0 = P(low volatility), Column 1 = P(high volatility)
        """
        if not self.is_fitted:
            # Return uniform probabilities if not fitted
            if returns.ndim == 1:
                n_samples = len(returns)
            else:
                n_samples = returns.shape[0]
            return np.full((n_samples, self.n_regimes), 1.0 / self.n_regimes)

        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Handle NaN/inf by replacing with 0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Get posterior probabilities using forward-backward algorithm
        try:
            posteriors = self.model.predict_proba(returns)
        except Exception:
            # Return uniform if prediction fails
            return np.full((returns.shape[0], self.n_regimes), 1.0 / self.n_regimes)

        # Reorder columns so that column 0 = low vol, column 1 = high vol
        return posteriors[:, self.regime_order]

    def get_current_regime(self, returns: np.ndarray) -> int:
        """
        Get the most likely regime for the most recent time step.

        Args:
            returns: Recent returns history

        Returns:
            0 = low volatility, 1 = high volatility
        """
        proba = self.predict_proba(returns)
        return int(np.argmax(proba[-1]))

    def get_regime_features(self, returns: np.ndarray, momentum_lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get regime features for RL observation space.

        Args:
            returns: Array of recent returns (minimum lookback + momentum_lookback)
            momentum_lookback: Number of steps to calculate regime momentum

        Returns:
            Tuple of:
                - regime_proba: (2,) array with [P(low_vol), P(high_vol)]
                - regime_momentum: (2,) array with regime transition speed
        """
        proba = self.predict_proba(returns)

        # Current regime probabilities (last time step)
        current_proba = proba[-1]  # Shape: (2,)

        # Regime momentum: change in probability over recent steps
        if len(proba) > momentum_lookback:
            past_mean_proba = proba[-momentum_lookback-1:-1].mean(axis=0)
            regime_momentum = current_proba - past_mean_proba
        else:
            regime_momentum = np.zeros(self.n_regimes)

        return current_proba, regime_momentum


def create_regime_detector(historical_data, lookback: int = 100) -> RegimeDetector:
    """
    Factory function to create and fit a RegimeDetector from historical data.

    Args:
        historical_data: DataFrame with 'close' or 'returns' column
        lookback: Lookback period for regime detection

    Returns:
        Fitted RegimeDetector instance
    """
    # Calculate returns if not present
    if 'returns' in historical_data.columns:
        returns = historical_data['returns'].values
    elif 'close' in historical_data.columns:
        prices = historical_data['close'].values
        returns = np.diff(np.log(prices))  # Log returns
    else:
        raise ValueError("Data must have 'returns' or 'close' column")

    # Create and fit detector
    detector = RegimeDetector(n_regimes=2, lookback=lookback)
    detector.fit(returns)

    return detector
