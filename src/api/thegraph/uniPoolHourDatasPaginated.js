import { getPoolHourData } from './uniPoolHourDatas';

/**
 * Fetches hourly pool data for extended periods (beyond 41 days) using pagination
 *
 * @param {string} poolId - The pool ID to fetch data for
 * @param {number} fromDate - Unix timestamp to start fetching from
 * @param {AbortSignal} signal - Abort signal for cancellation
 * @param {number} protocolId - Protocol identifier (0=Ethereum, 1=Optimism, etc.)
 * @param {function} onProgress - Callback function called after each chunk (receives current chunk count)
 * @returns {Promise<Array>} - Array of hourly pool data records, sorted by time
 */
export const getPoolHourDataPaginated = async (
  poolId,
  fromDate,
  signal,
  protocolId,
  onProgress
) => {
  const CHUNK_SIZE_DAYS = 40; // 40 days = ~960 records (safe margin under 1000 limit)
  const CHUNK_SIZE_SECONDS = CHUNK_SIZE_DAYS * 24 * 60 * 60;

  const chunks = [];
  let currentDate = fromDate;
  const now = Math.round(Date.now() / 1000);

  // Debug: Log date range being fetched
  console.log(`[Pagination] Fetching data from ${new Date(fromDate * 1000).toISOString()} to ${new Date(now * 1000).toISOString()}`);
  console.log(`[Pagination] Days requested: ${Math.round((now - fromDate) / 86400)} days`);

  // Fetch data in chunks
  while (currentDate < now && !signal.aborted) {
    const chunkEndDate = Math.min(
      currentDate + CHUNK_SIZE_SECONDS,
      now
    );

    console.log(`[Pagination] Chunk ${chunks.length + 1}: ${new Date(currentDate * 1000).toISOString()} to ${new Date(chunkEndDate * 1000).toISOString()}`);

    try {
      // Fetch individual chunk with toDate limit
      const chunk = await getPoolHourData(
        poolId,
        currentDate,
        signal,
        protocolId,
        chunkEndDate  // Add toDate parameter to limit data range
      );

      if (!chunk || chunk.error) {
        // Check if this is an abort error
        const isAbortError = chunk && chunk.error && chunk.error.name === 'AbortError';

        if (isAbortError) {
          console.log(`[Pagination] Chunk aborted for date range: ${currentDate} to ${chunkEndDate}`);
          break;
        }

        console.warn(`Chunk failed for date range: ${currentDate} to ${chunkEndDate}`, chunk ? chunk.error : 'No data');

        // If first chunk fails, throw error - cannot proceed
        if (chunks.length === 0) {
          throw new Error('First chunk failed - cannot proceed');
        }

        // Otherwise, use partial data
        break;
      }

      console.log(`[Pagination] Chunk ${chunks.length + 1} received ${chunk.length} records`);

      // Safety check: if we're stuck in a loop (getting very few records repeatedly)
      if (chunk.length === 0) {
        console.warn('[Pagination] Empty chunk received, stopping pagination');
        break;
      }

      chunks.push(chunk);

      // Call progress callback if provided
      if (onProgress) {
        onProgress(chunks.length);
      }

      // Move to next chunk (no overlap needed since we use gte/lt which are non-overlapping)
      currentDate = chunkEndDate;

    } catch (error) {
      console.error('Error fetching chunk:', error);

      // If first chunk fails, rethrow
      if (chunks.length === 0) {
        throw error;
      }

      // Otherwise, break and use what we have
      break;
    }
  }

  // Handle abort
  if (signal.aborted) {
    console.log('Fetch aborted by user');
    // Return whatever we've fetched so far
  }

  // Flatten, deduplicate, and sort
  return deduplicateAndSort(chunks.flat());
};

/**
 * Deduplicates records by periodStartUnix and sorts chronologically (descending - newest first)
 *
 * @param {Array} records - Array of pool hour data records
 * @returns {Array} - Deduplicated and sorted records (newest first, matching getPoolHourData behavior)
 */
function deduplicateAndSort(records) {
  const uniqueMap = new Map();

  // Keep only the first occurrence of each timestamp
  records.forEach(record => {
    const key = record.periodStartUnix;
    if (!uniqueMap.has(key)) {
      uniqueMap.set(key, record);
    }
  });

  // Convert back to array and sort by timestamp (descending - newest first)
  // This matches the behavior of getPoolHourData which returns desc order
  return Array.from(uniqueMap.values())
    .sort((a, b) => b.periodStartUnix - a.periodStartUnix);
}
