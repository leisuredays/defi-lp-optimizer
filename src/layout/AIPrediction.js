import React, { useEffect, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import styles from '../styles/modules/AIPrediction.module.css';
import { selectPoolID, selectBaseToken, selectQuoteToken, selectBaseTokenId } from '../store/pool';
import { selectProtocolId } from '../store/protocol';
import { selectInvestment } from '../store/investment';
import {
  fetchAvailableModels,
  getPrediction,
  runBacktest,
  setSelectedModel,
  clearPrediction,
  clearBacktest,
  selectModels,
  selectModelsLoading,
  selectModelsError,
  selectSelectedModel,
  selectPrediction,
  selectPredictionLoading,
  selectPredictionError,
  selectBacktest,
  selectBacktestLoading,
  selectBacktestError,
  selectLoadingProgress,
  selectBacktestDays,
  ErrorTypes
} from '../store/prediction';
import RebalancingTrendChart from '../components/uniswap/RebalancingTrendChart';

// Skeleton Loading Component
const SkeletonLoader = ({ width = '100%', height = '20px', className = '' }) => (
  <div
    className={`${styles['skeleton']} ${className}`}
    style={{ width, height }}
  />
);

// Skeleton Section Component
const SkeletonSection = ({ title, rows = 3 }) => (
  <div className={styles['section']}>
    <h4 className={styles['section-title']}>{title}</h4>
    <div className={styles['info-grid']}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className={styles['info-item']}>
          <SkeletonLoader width="40%" height="14px" />
          <SkeletonLoader width="30%" height="14px" />
        </div>
      ))}
    </div>
  </div>
);

// Error Display Component
const ErrorDisplay = ({ error, onRetry }) => {
  const getErrorIcon = (type) => {
    switch (type) {
      case ErrorTypes.NETWORK:
        return '🔌';
      case ErrorTypes.TIMEOUT:
        return '⏱️';
      case ErrorTypes.SERVER:
        return '🖥️';
      case ErrorTypes.MODEL_NOT_FOUND:
        return '🔍';
      case ErrorTypes.POOL_NOT_FOUND:
        return '💧';
      case ErrorTypes.VALIDATION:
        return '📝';
      default:
        return '⚠️';
    }
  };

  const getErrorAction = (type) => {
    switch (type) {
      case ErrorTypes.NETWORK:
        return 'Check if the backend server is running on the correct port.';
      case ErrorTypes.TIMEOUT:
        return 'The request took too long. Try again or select a different pool.';
      case ErrorTypes.SERVER:
        return 'There was a server-side issue. Please try again later.';
      case ErrorTypes.MODEL_NOT_FOUND:
        return 'Select a different model or check model availability.';
      case ErrorTypes.POOL_NOT_FOUND:
        return 'This pool may not exist on the selected network.';
      default:
        return 'Please try again.';
    }
  };

  return (
    <div className={styles['error-container']}>
      <div className={styles['error']}>
        <span className={styles['error-icon']}>{getErrorIcon(error?.type)}</span>
        <div className={styles['error-content']}>
          <span className={styles['error-message']}>{error?.message || 'An error occurred'}</span>
          <span className={styles['error-action']}>{getErrorAction(error?.type)}</span>
        </div>
      </div>
      {onRetry && (
        <button className={styles['retry-btn']} onClick={onRetry}>
          🔄 Retry
        </button>
      )}
    </div>
  );
};

// Progress Bar Component
const ProgressBar = ({ progress }) => (
  <div className={styles['progress-container']}>
    <div className={styles['progress-bar']}>
      <div
        className={styles['progress-fill']}
        style={{ width: `${(progress.current / progress.total) * 100}%` }}
      />
    </div>
    <span className={styles['progress-text']}>{progress.message}</span>
  </div>
);

const AIPrediction = (props) => {
  const dispatch = useDispatch();

  // Pool/Protocol selectors
  const poolID = useSelector(selectPoolID);
  const baseToken = useSelector(selectBaseToken);
  const quoteToken = useSelector(selectQuoteToken);
  const baseTokenId = useSelector(selectBaseTokenId);
  const protocolId = useSelector(selectProtocolId);
  const investment = useSelector(selectInvestment);

  // Prediction selectors
  const models = useSelector(selectModels);
  const modelsLoading = useSelector(selectModelsLoading);
  const modelsError = useSelector(selectModelsError);
  const selectedModel = useSelector(selectSelectedModel);
  const prediction = useSelector(selectPrediction);
  const loading = useSelector(selectPredictionLoading);
  const error = useSelector(selectPredictionError);
  const loadingProgress = useSelector(selectLoadingProgress);

  // Backtest selectors
  const backtest = useSelector(selectBacktest);
  const backtestLoading = useSelector(selectBacktestLoading);
  const backtestError = useSelector(selectBacktestError);
  const backtestDays = useSelector(selectBacktestDays);

  // Fetch available models on mount
  useEffect(() => {
    dispatch(fetchAvailableModels());
  }, [dispatch]);

  // Clear prediction and backtest when pool changes
  useEffect(() => {
    dispatch(clearPrediction());
    dispatch(clearBacktest());
  }, [poolID, dispatch]);

  // Transform prediction data based on baseToken selection
  // Backend returns token1/token0 price, need to invert if baseTokenId === 1
  const transformedPrediction = useMemo(() => {
    if (!prediction) return null;

    // If baseTokenId is 1, user has toggled base token, need to invert prices
    const shouldInvert = baseTokenId === 1;

    const transformPrice = (price) => {
      if (!price || price === 0) return price;
      return shouldInvert ? 1 / price : price;
    };

    return {
      ...prediction,
      pool_info: {
        ...prediction.pool_info,
        // Display as quoteToken/baseToken (same as existing UI)
        token0: shouldInvert ? prediction.pool_info?.token1 : prediction.pool_info?.token0,
        token1: shouldInvert ? prediction.pool_info?.token0 : prediction.pool_info?.token1,
        current_price: transformPrice(prediction.pool_info?.current_price)
      },
      current_state: {
        ...prediction.current_state,
        price: transformPrice(prediction.current_state?.price),
        volatility: shouldInvert && prediction.current_state?.price
          ? prediction.current_state.volatility / (prediction.current_state.price * prediction.current_state.price)
          : prediction.current_state?.volatility
      },
      predicted_range: {
        ...prediction.predicted_range,
        // When inverting, min becomes 1/max and max becomes 1/min
        min: shouldInvert
          ? transformPrice(prediction.predicted_range?.max)
          : prediction.predicted_range?.min,
        max: shouldInvert
          ? transformPrice(prediction.predicted_range?.min)
          : prediction.predicted_range?.max
      }
    };
  }, [prediction, baseTokenId]);

  const handlePredict = () => {
    if (!poolID || typeof protocolId !== 'number') {
      return;
    }

    dispatch(getPrediction({
      poolId: poolID,
      protocolId: protocolId,
      modelName: selectedModel,
      investment: investment || 10000
    }));
  };

  const handleBacktest = () => {
    console.log('[AIPrediction] handleBacktest called', { selectedModel, poolID, protocolId, backtestDays });

    if (!selectedModel) {
      console.error('[AIPrediction] No model selected');
      return;
    }

    if (!poolID) {
      console.error('[AIPrediction] No pool selected');
      return;
    }

    if (typeof protocolId !== 'number') {
      console.error('[AIPrediction] Invalid protocol ID');
      return;
    }

    // Now using live data - support up to 30 days
    const maxDays = 30;
    const effectiveDays = Math.min(backtestDays, maxDays);

    console.log('[AIPrediction] Dispatching runBacktest', {
      poolId: poolID,
      protocolId: protocolId,
      modelName: selectedModel,
      episodeLengthDays: effectiveDays
    });

    dispatch(runBacktest({
      poolId: poolID,
      protocolId: protocolId,
      modelName: selectedModel,
      nEpisodes: 1,
      episodeLengthDays: effectiveDays,
      debug: true  // Enable debug logging
    }));
  };

  const handleModelChange = (e) => {
    dispatch(setSelectedModel(e.target.value));
  };

  const handleRetry = () => {
    if (modelsError) {
      dispatch(fetchAvailableModels());
    } else {
      handlePredict();
    }
  };

  const formatPrice = (price) => {
    if (!price) return 'N/A';
    return price.toFixed(6);
  };

  const formatUSD = (amount) => {
    if (!amount) return '$0.00';
    return `$${amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercent = (percent) => {
    if (!percent) return '0.00%';
    return `${percent.toFixed(2)}%`;
  };

  return (
    <div className={`${styles['ai-prediction']} ${props.pageStyle?.['outer-glow'] || ''}`}>
      <div className={styles['header']}>
        <h3 className={styles['title']}>
          🤖 AI Range Prediction
          {transformedPrediction && (
            <span className={styles['version-badge']}>
              V2
            </span>
          )}
        </h3>
      </div>

      <div className={styles['content']}>
        {/* Model Selection */}
        <div className={styles['model-selector']}>
          <label className={styles['label']}>Model:</label>
          {modelsLoading ? (
            <SkeletonLoader height="40px" />
          ) : modelsError ? (
            <ErrorDisplay error={modelsError} onRetry={handleRetry} />
          ) : (
            <select
              className={styles['select']}
              value={selectedModel || ''}
              onChange={handleModelChange}
              disabled={loading || models.length === 0}
            >
              {models.length === 0 ? (
                <option value="">No models available</option>
              ) : (
                models.map(model => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                    {model.observation_dim && ` (${model.observation_dim}D)`}
                    {' '}({model.size_mb} MB)
                  </option>
                ))
              )}
            </select>
          )}
        </div>

        {/* Action Buttons */}
        <div className={styles['button-group']}>
          <button
            className={`${styles['predict-btn']} ${loading ? styles['loading'] : ''}`}
            onClick={handlePredict}
            disabled={loading || backtestLoading || !poolID || !selectedModel}
          >
            {loading ? '⏳ Predicting...' : '🚀 Get AI Prediction'}
          </button>
          <button
            className={`${styles['backtest-btn']} ${backtestLoading ? styles['loading'] : ''}`}
            onClick={handleBacktest}
            disabled={loading || backtestLoading || !selectedModel}
          >
            {backtestLoading ? '⏳ Running Backtest...' : '📊 Run Test Validation'}
          </button>
        </div>

        {/* Loading Progress */}
        {loading && loadingProgress.total > 0 && (
          <ProgressBar progress={loadingProgress} />
        )}

        {/* Skeleton Loading State */}
        {loading && (
          <div className={styles['results']}>
            <SkeletonSection title="🧠 Model Architecture" rows={4} />
            <SkeletonSection title="⚖️ Reward Function" rows={3} />
            <SkeletonSection title="📊 Input State (24 Features)" rows={6} />
            <SkeletonSection title="Pool Information" rows={3} />
            <div className={`${styles['section']} ${styles['highlight']}`}>
              <h4 className={styles['section-title']}>✨ Predicted Optimal Range</h4>
              <div className={styles['range-box']}>
                <SkeletonLoader width="100%" height="24px" />
                <SkeletonLoader width="100%" height="24px" />
                <SkeletonLoader width="60%" height="24px" />
              </div>
            </div>
            <SkeletonSection title="📈 Expected Performance" rows={5} />
          </div>
        )}

        {/* Error Message */}
        {error && !loading && (
          <ErrorDisplay error={error} onRetry={handleRetry} />
        )}

        {/* Prediction Results */}
        {transformedPrediction && !loading && (
          <div className={styles['results']}>
            {/* Model Architecture */}
            <div className={styles['section']}>
              <h4 className={styles['section-title']}>🧠 Model Architecture</h4>
              <div className={styles['architecture']}>
                <div className={styles['arch-layer']}>
                  <span className={styles['arch-label']}>Input Layer:</span>
                  <span className={styles['arch-value']}>
                    {transformedPrediction.model_metadata?.architecture?.input_dim || 24} features
                  </span>
                </div>
                <div className={styles['arch-arrow']}>↓</div>
                {(transformedPrediction.model_metadata?.architecture?.hidden_layers || [
                  { neurons: 256, activation: 'SiLU' },
                  { neurons: 256, activation: 'SiLU' }
                ]).map((layer, idx) => (
                  <React.Fragment key={idx}>
                    <div className={styles['arch-layer']}>
                      <span className={styles['arch-label']}>Hidden Layer {idx + 1}:</span>
                      <span className={styles['arch-value']}>
                        {layer.neurons} neurons ({layer.activation})
                      </span>
                    </div>
                    <div className={styles['arch-arrow']}>↓</div>
                  </React.Fragment>
                ))}
                <div className={styles['arch-layer']}>
                  <span className={styles['arch-label']}>Output Layer:</span>
                  <span className={styles['arch-value']}>
                    {transformedPrediction.model_metadata?.architecture?.output_dim || 3} actions (rebalance + range)
                  </span>
                </div>
              </div>
              <div className={styles['arch-info']}>
                ✨ V2: Log-scale mapping, sigmoid rebalancing, reduced features
              </div>
            </div>

            {/* Reward Function */}
            <div className={styles['section']}>
              <h4 className={styles['section-title']}>⚖️ Reward Function</h4>
              <div className={styles['reward-formula']}>
                <code>{transformedPrediction.model_metadata?.reward_function?.formula || 'reward = α×fees - β×IL - γ×gas'}</code>
              </div>
              <div className={styles['info-grid']}>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>α (Fees weight):</span>
                  <span className={styles['info-value']}>
                    {transformedPrediction.model_metadata?.reward_function?.weights?.alpha || 1.0}
                  </span>
                </div>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>β (IL weight):</span>
                  <span className={styles['info-value']}>
                    {transformedPrediction.model_metadata?.reward_function?.weights?.beta || 0.8}
                  </span>
                </div>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>γ (Gas weight):</span>
                  <span className={styles['info-value']}>
                    {transformedPrediction.model_metadata?.reward_function?.weights?.gamma || 0.2}
                  </span>
                </div>
              </div>
            </div>

            {/* Input State Features */}
            <div className={styles['section']}>
              <h4 className={styles['section-title']}>📊 Input State (24 Features)</h4>
              <div className={styles['state-features']}>
                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Price Information (4)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Current Price:</span>
                    <span className={styles['feature-value']}>{formatPrice(transformedPrediction.current_state?.price)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>24h Avg Price:</span>
                    <span className={styles['feature-value']}>~{formatPrice(transformedPrediction.current_state?.price)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>7d Avg Price:</span>
                    <span className={styles['feature-value']}>~{formatPrice(transformedPrediction.current_state?.price)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Volatility:</span>
                    <span className={styles['feature-value']}>{formatPrice(transformedPrediction.current_state?.volatility)}</span>
                  </div>
                </div>

                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Liquidity Information (3)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Pool Liquidity:</span>
                    <span className={styles['feature-value']}>{formatUSD(transformedPrediction.current_state?.liquidity)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Position Liquidity:</span>
                    <span className={styles['feature-value']}>{formatUSD(investment)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Liquidity Ratio:</span>
                    <span className={styles['feature-value']}>
                      {transformedPrediction.current_state?.liquidity
                        ? ((investment / transformedPrediction.current_state.liquidity) * 100).toFixed(4)
                        : '0.0000'}%
                    </span>
                  </div>
                </div>

                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Position Information (6)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Initial Min Price:</span>
                    <span className={styles['feature-value']}>{formatPrice(transformedPrediction.predicted_range?.min * 0.95)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Initial Max Price:</span>
                    <span className={styles['feature-value']}>{formatPrice(transformedPrediction.predicted_range?.max * 1.05)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>In-Range %:</span>
                    <span className={styles['feature-value']}>100%</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Position Value:</span>
                    <span className={styles['feature-value']}>{formatUSD(investment)}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Cumulative Fees:</span>
                    <span className={styles['feature-value']}>$0</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Cumulative IL:</span>
                    <span className={styles['feature-value']}>$0</span>
                  </div>
                </div>

                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Market Indicators (6)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>24h Volume:</span>
                    <span className={styles['feature-value']}>Live data</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Market Volatility:</span>
                    <span className={styles['feature-value']}>
                      {transformedPrediction.current_state?.price
                        ? formatPercent((transformedPrediction.current_state.volatility / transformedPrediction.current_state.price) * 100)
                        : '0.00%'}
                    </span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Trend Indicator:</span>
                    <span className={styles['feature-value']}>Calculated</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>RSI:</span>
                    <span className={styles['feature-value']}>50.0</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Bollinger Upper:</span>
                    <span className={styles['feature-value']}>Calculated</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Bollinger Lower:</span>
                    <span className={styles['feature-value']}>Calculated</span>
                  </div>
                </div>

                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Return Metrics (4)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Current Return %:</span>
                    <span className={styles['feature-value']}>0%</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>APR:</span>
                    <span className={styles['feature-value']}>0%</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Sharpe Ratio:</span>
                    <span className={styles['feature-value']}>0.0</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Rebalances:</span>
                    <span className={styles['feature-value']}>0</span>
                  </div>
                </div>

                <div className={styles['feature-group']}>
                  <h5 className={styles['feature-group-title']}>Time Information (5)</h5>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Episode Progress:</span>
                    <span className={styles['feature-value']}>0%</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Elapsed Hours:</span>
                    <span className={styles['feature-value']}>0</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Time of Day:</span>
                    <span className={styles['feature-value']}>{new Date().getHours()}:00</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Day of Week:</span>
                    <span className={styles['feature-value']}>{['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][new Date().getDay()]}</span>
                  </div>
                  <div className={styles['feature-item']}>
                    <span className={styles['feature-label']}>Hour:</span>
                    <span className={styles['feature-value']}>{new Date().getHours()}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Pool Info */}
            <div className={styles['section']}>
              <h4 className={styles['section-title']}>Pool Information</h4>
              <div className={styles['info-grid']}>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>Pair:</span>
                  <span className={styles['info-value']}>
                    {quoteToken?.symbol}/{baseToken?.symbol}
                  </span>
                </div>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>Fee Tier:</span>
                  <span className={styles['info-value']}>
                    {transformedPrediction.pool_info?.fee_tier / 10000}%
                  </span>
                </div>
                <div className={styles['info-item']}>
                  <span className={styles['info-label']}>Current Price:</span>
                  <span className={styles['info-value']}>
                    {formatPrice(transformedPrediction.pool_info?.current_price)} {baseToken?.symbol}
                  </span>
                </div>
              </div>
            </div>

            {/* Predicted Range */}
            <div className={`${styles['section']} ${styles['highlight']}`}>
              <h4 className={styles['section-title']}>
                ✨ Predicted Optimal Range
              </h4>
              <div className={styles['range-box']}>
                <div className={styles['range-item']}>
                  <span className={styles['range-label']}>Min Price:</span>
                  <span className={styles['range-value']}>
                    {formatPrice(transformedPrediction.predicted_range?.min)} {baseToken?.symbol}
                  </span>
                </div>
                <div className={styles['range-item']}>
                  <span className={styles['range-label']}>Max Price:</span>
                  <span className={styles['range-value']}>
                    {formatPrice(transformedPrediction.predicted_range?.max)} {baseToken?.symbol}
                  </span>
                </div>
                <div className={styles['range-item']}>
                  <span className={styles['range-label']}>Confidence:</span>
                  <span className={`${styles['range-value']} ${styles['confidence']}`}>
                    {formatPercent(transformedPrediction.predicted_range?.confidence * 100)}
                  </span>
                </div>
              </div>
            </div>

            {/* Model Info */}
            <div className={styles['footer']}>
              <span className={styles['footer-text']}>
                Model: {transformedPrediction.model_name} v{transformedPrediction.model_version}
              </span>
              <span className={styles['footer-text']}>
                Updated: {new Date(transformedPrediction.timestamp).toLocaleString()}
              </span>
            </div>
          </div>
        )}

        {/* Backtest Error */}
        {backtestError && !backtestLoading && (
          <ErrorDisplay error={backtestError} onRetry={handleBacktest} />
        )}

        {/* Backtest Loading */}
        {backtestLoading && (
          <div className={styles['section']}>
            <h4 className={styles['section-title']}>📊 Running Test Validation...</h4>
            <ProgressBar progress={loadingProgress} />
            <div className={styles['info-grid']}>
              <SkeletonLoader width="100%" height="200px" />
            </div>
          </div>
        )}

        {/* Rebalancing Trend Chart */}
        {backtest && !backtestLoading && (
          <RebalancingTrendChart backtestData={backtest} height={280} />
        )}

        {/* Empty State */}
        {!prediction && !backtest && !loading && !backtestLoading && !error && !backtestError && (
          <div className={styles['empty-state']}>
            <div className={styles['empty-icon']}>🎯</div>
            <p className={styles['empty-text']}>
              Select a pool and click "Get AI Prediction" to see optimal LP ranges powered by machine learning.
              <br /><br />
              Or click "Run Test Validation" to see how the model performs on historical test data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIPrediction;
