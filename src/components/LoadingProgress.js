import React from 'react';
import styles from '../styles/modules/LoadingProgress.module.css';

/**
 * LoadingProgress - Displays progress when fetching multiple data chunks
 *
 * @param {number} current - Current chunk number
 * @param {number} total - Total number of chunks to fetch
 * @param {string} className - Additional CSS class names
 */
const LoadingProgress = ({ current, total, className }) => {
  // Don't show progress bar if there's only one chunk (instant load)
  if (total === 0 || total === 1) {
    return null;
  }

  const percentage = Math.min(100, (current / total) * 100);

  return (
    <div className={`${styles['loading-progress']} ${className || ''}`}>
      <div
        className={styles['progress-bar']}
        style={{ width: `${percentage}%` }}
      />
      <span className={styles['progress-text']}>
        Loading data: {current}/{total} chunks ({percentage.toFixed(0)}%)
      </span>
    </div>
  );
};

export default LoadingProgress;
