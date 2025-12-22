import { useState, useRef, useEffect, useMemo } from 'react';
import { useSelector } from 'react-redux';
import { scaleLinear, scaleTime } from 'd3-scale';
import { line, curveStepAfter } from 'd3-shape';
import { axisBottom, axisLeft } from 'd3-axis';
import { select } from 'd3-selection';
import { selectBaseTokenId } from '../../store/pool';
import styles from '../../styles/modules/RebalancingTrendChart.module.css';

/**
 * RebalancingTrendChart Component
 *
 * Displays the price history with liquidity range bounds and rebalancing events.
 * Visualization matches the user's sketch:
 * - Price line (black/white smooth curve)
 * - LP Range bounds (red horizontal step lines - change only at rebalancing)
 * - Rebalancing events (blue vertical lines with dots)
 */
const RebalancingTrendChart = ({ backtestData, height = 300 }) => {
  const containerRef = useRef();
  const svgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const baseTokenId = useSelector(selectBaseTokenId);

  const margin = { top: 20, right: 70, bottom: 40, left: 70 };

  // Process and transform data based on baseTokenId
  const chartData = useMemo(() => {
    if (!backtestData?.episodes?.[0]?.events) return null;

    const events = backtestData.episodes[0].events;
    const shouldInvert = baseTokenId === 1;

    return events.map(event => ({
      timestamp: new Date(event.timestamp * 1000),
      price: shouldInvert ? 1 / event.price : event.price,
      minPrice: shouldInvert ? 1 / event.max_price : event.min_price,
      maxPrice: shouldInvert ? 1 / event.min_price : event.max_price,
      rebalanced: event.rebalanced,
      inRange: event.in_range,
      hour: event.hour
    }));
  }, [backtestData, baseTokenId]);

  // Rebalancing events (filtered)
  const rebalanceEvents = useMemo(() => {
    if (!chartData) return [];
    return chartData.filter(d => d.rebalanced);
  }, [chartData]);

  // Calculate domains - focus on price range for better visibility
  const domains = useMemo(() => {
    if (!chartData || chartData.length === 0) return null;

    const timestamps = chartData.map(d => d.timestamp);
    const allPrices = chartData.flatMap(d => [d.price, d.minPrice, d.maxPrice]);

    // Filter out extreme values for better visualization
    const validPrices = allPrices.filter(p => p > 0 && isFinite(p));
    const priceMean = validPrices.reduce((a, b) => a + b, 0) / validPrices.length;
    const reasonablePrices = validPrices.filter(p => p < priceMean * 10 && p > priceMean * 0.1);

    const minPrice = Math.min(...reasonablePrices) * 0.95;
    const maxPrice = Math.max(...reasonablePrices) * 1.05;

    return {
      x: [timestamps[0], timestamps[timestamps.length - 1]],
      y: [minPrice, maxPrice]
    };
  }, [chartData]);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: width - margin.left - margin.right,
          height: height - margin.top - margin.bottom
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [height, margin.left, margin.right, margin.top, margin.bottom]);

  // Render chart
  useEffect(() => {
    if (!chartData || !domains || dimensions.width <= 0) return;

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height: chartHeight } = dimensions;

    // Create scales
    const xScale = scaleTime()
      .domain(domains.x)
      .range([0, width]);

    const yScale = scaleLinear()
      .domain(domains.y)
      .range([chartHeight, 0]);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // === 1. LP Range Bounds (Red Step Lines) ===
    // Min price line (step-like - changes only at rebalancing)
    const minLine = line()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.minPrice))
      .curve(curveStepAfter);  // Step function for horizontal segments

    g.append('path')
      .datum(chartData)
      .attr('class', styles.minRangeLine)
      .attr('d', minLine);

    // Max price line (step-like)
    const maxLine = line()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.maxPrice))
      .curve(curveStepAfter);

    g.append('path')
      .datum(chartData)
      .attr('class', styles.maxRangeLine)
      .attr('d', maxLine);

    // Price labels for LP Range bounds (right side)
    const formatRangePrice = (price) => {
      if (price >= 1000) return `$${(price/1000).toFixed(2)}K`;
      if (price >= 1) return `$${price.toFixed(2)}`;
      if (price >= 0.01) return `$${price.toFixed(4)}`;
      return `$${price.toExponential(2)}`;
    };

    const lastData = chartData[chartData.length - 1];

    // Min price label (bottom red line)
    g.append('text')
      .attr('class', styles.rangePriceLabel)
      .attr('x', width + 5)
      .attr('y', yScale(lastData.minPrice))
      .attr('dy', '0.35em')
      .style('fill', '#ef4444')
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .text(formatRangePrice(lastData.minPrice));

    // Max price label (top red line)
    g.append('text')
      .attr('class', styles.rangePriceLabel)
      .attr('x', width + 5)
      .attr('y', yScale(lastData.maxPrice))
      .attr('dy', '0.35em')
      .style('fill', '#ef4444')
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .text(formatRangePrice(lastData.maxPrice));

    // Current price label (at the end of price line)
    g.append('text')
      .attr('class', styles.currentPriceLabel)
      .attr('x', width + 5)
      .attr('y', yScale(lastData.price))
      .attr('dy', '0.35em')
      .style('fill', '#6b7280')
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .text(formatRangePrice(lastData.price));

    // === 2. Price Line (White/Black Smooth Curve) ===
    const priceLine = line()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.price));

    g.append('path')
      .datum(chartData)
      .attr('class', styles.priceLine)
      .attr('d', priceLine);

    // === 3. Rebalancing Events (Blue Vertical Lines with Dots) ===
    rebalanceEvents.forEach(event => {
      const x = xScale(event.timestamp);

      // Vertical line
      g.append('line')
        .attr('class', styles.rebalanceLine)
        .attr('x1', x)
        .attr('x2', x)
        .attr('y1', 0)
        .attr('y2', chartHeight);

      // Circle marker at price point
      g.append('circle')
        .attr('class', styles.rebalanceMarker)
        .attr('cx', x)
        .attr('cy', yScale(event.price))
        .attr('r', 6);
    });

    // === 4. Axes ===
    const xAxis = axisBottom(xScale)
      .ticks(6)
      .tickFormat(d => {
        const date = new Date(d);
        return `${date.getMonth() + 1}/${date.getDate()}`;
      });

    g.append('g')
      .attr('class', styles.axis)
      .attr('transform', `translate(0,${chartHeight})`)
      .call(xAxis);

    const yAxis = axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => {
        if (d >= 1000) return `${(d/1000).toFixed(1)}K`;
        if (d >= 1) return d.toFixed(2);
        if (d >= 0.01) return d.toFixed(4);
        return d.toExponential(2);
      });

    g.append('g')
      .attr('class', styles.axis)
      .call(yAxis);

    // Y axis label
    g.append('text')
      .attr('class', styles.axisLabel)
      .attr('transform', 'rotate(-90)')
      .attr('x', -chartHeight / 2)
      .attr('y', -55)
      .attr('text-anchor', 'middle')
      .text('Price');

  }, [chartData, domains, dimensions, rebalanceEvents, margin]);

  if (!backtestData) {
    return (
      <div className={styles.container} ref={containerRef}>
        <div className={styles.noData}>
          Run backtest to see rebalancing trend
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container} ref={containerRef}>
      <div className={styles.header}>
        <h3 className={styles.title}>Rebalancing History</h3>
        <div className={styles.legend}>
          <span className={styles.legendItem}>
            <span className={styles.priceLegendLine}></span> Price
          </span>
          <span className={styles.legendItem}>
            <span className={styles.rangeLegendLine}></span> LP Range
          </span>
          <span className={styles.legendItem}>
            <span className={styles.rebalanceLegendDot}></span> Rebalance
          </span>
        </div>
      </div>
      <svg
        ref={svgRef}
        width={dimensions.width + margin.left + margin.right}
        height={height}
      />
      <div className={styles.stats}>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Total Rebalances</span>
          <span className={styles.statValue}>{rebalanceEvents.length}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Test Period</span>
          <span className={styles.statValue}>{backtestData.test_period_days} days</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Total Fees</span>
          <span className={`${styles.statValue} ${styles.positive}`}>
            ${backtestData.episodes?.[0]?.total_fees?.toFixed(2) || '0.00'}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Total IL</span>
          <span className={`${styles.statValue} ${styles.negative}`}>
            ${backtestData.episodes?.[0]?.total_il?.toFixed(2) || '0.00'}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Total Gas</span>
          <span className={`${styles.statValue} ${styles.negative}`}>
            ${backtestData.episodes?.[0]?.total_gas?.toFixed(2) || '0.00'}
          </span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Net Return</span>
          <span className={`${styles.statValue} ${backtestData.summary?.avg_net_return >= 0 ? styles.positive : styles.negative}`}>
            ${backtestData.summary?.avg_net_return?.toFixed(2) || '0.00'}
          </span>
        </div>
      </div>
      <div className={styles.formula}>
        <span className={styles.formulaText}>
          Net Return = Fees - IL - Gas = ${backtestData.episodes?.[0]?.total_fees?.toFixed(2) || '0'} - ${backtestData.episodes?.[0]?.total_il?.toFixed(2) || '0'} - ${backtestData.episodes?.[0]?.total_gas?.toFixed(2) || '0'} = ${backtestData.summary?.avg_net_return?.toFixed(2) || '0'}
        </span>
      </div>

      {/* Investment Summary Section */}
      {backtestData.episodes?.[0]?.events?.length > 0 && (() => {
        const initialEvent = backtestData.episodes[0].events[0];
        const finalEvent = backtestData.episodes[0].events[backtestData.episodes[0].events.length - 1];
        const token0Symbol = backtestData.episodes[0].token0_symbol || 'Token0';
        const token1Symbol = backtestData.episodes[0].token1_symbol || 'Token1';

        const initialInvestment = initialEvent.position_value || 0;
        const finalValue = finalEvent.position_value || 0;
        const profitLoss = finalValue - initialInvestment;
        const profitLossPercent = initialInvestment > 0 ? (profitLoss / initialInvestment * 100) : 0;

        return (
          <>
            {/* Initial Investment */}
            <div className={styles.tokenHoldings}>
              <div className={styles.tokenHeader}>Initial Investment:</div>
              <div className={styles.tokenStats}>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>{token0Symbol}</span>
                  <span className={styles.tokenValue}>
                    {initialEvent.token0_amount?.toFixed(6) || '0.000000'}
                    <span className={styles.tokenValueUsd}>(${initialEvent.token0_value?.toFixed(2) || '0.00'})</span>
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>{token1Symbol}</span>
                  <span className={styles.tokenValue}>
                    {initialEvent.token1_amount?.toFixed(6) || '0.000000'}
                    <span className={styles.tokenValueUsd}>(${initialEvent.token1_value?.toFixed(2) || '0.00'})</span>
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>Total Investment</span>
                  <span className={`${styles.tokenValue} ${styles.totalValue}`}>
                    ${initialInvestment.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>

            {/* Final Position */}
            <div className={styles.tokenHoldings}>
              <div className={styles.tokenHeader}>Final Position Holdings:</div>
              <div className={styles.tokenStats}>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>{token0Symbol}</span>
                  <span className={styles.tokenValue}>
                    {finalEvent.token0_amount?.toFixed(6) || '0.000000'}
                    <span className={styles.tokenValueUsd}>(${finalEvent.token0_value?.toFixed(2) || '0.00'})</span>
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>{token1Symbol}</span>
                  <span className={styles.tokenValue}>
                    {finalEvent.token1_amount?.toFixed(6) || '0.000000'}
                    <span className={styles.tokenValueUsd}>(${finalEvent.token1_value?.toFixed(2) || '0.00'})</span>
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>Final Value</span>
                  <span className={`${styles.tokenValue} ${styles.totalValue}`}>
                    ${finalValue.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>

            {/* Profit/Loss Summary */}
            <div className={styles.tokenHoldings}>
              <div className={styles.tokenHeader}>Performance Summary:</div>
              <div className={styles.tokenStats}>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>Profit/Loss</span>
                  <span className={`${styles.tokenValue} ${profitLoss >= 0 ? styles.profitPositive : styles.profitNegative}`}>
                    ${profitLoss >= 0 ? '+' : ''}{profitLoss.toFixed(2)}
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>Return %</span>
                  <span className={`${styles.tokenValue} ${profitLoss >= 0 ? styles.profitPositive : styles.profitNegative}`}>
                    {profitLoss >= 0 ? '+' : ''}{profitLossPercent.toFixed(2)}%
                  </span>
                </div>
                <div className={styles.tokenItem}>
                  <span className={styles.tokenLabel}>Rebalances</span>
                  <span className={styles.tokenValue}>
                    {backtestData.episodes[0].events.length - 1}
                  </span>
                </div>
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
};

export default RebalancingTrendChart;
