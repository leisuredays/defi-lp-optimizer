import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";

// API Base URL from environment variable
const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// Error types for categorization
export const ErrorTypes = {
  NETWORK: 'NETWORK',
  TIMEOUT: 'TIMEOUT',
  SERVER: 'SERVER',
  VALIDATION: 'VALIDATION',
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  POOL_NOT_FOUND: 'POOL_NOT_FOUND',
  UNKNOWN: 'UNKNOWN'
};

// Categorize error based on response or exception
const categorizeError = (error, response) => {
  if (!response) {
    if (error.name === 'AbortError') {
      return { type: ErrorTypes.TIMEOUT, message: 'Request timed out. Please try again.' };
    }
    if (error.message?.includes('fetch') || error.message?.includes('network')) {
      return { type: ErrorTypes.NETWORK, message: 'Network error. Is the backend server running?' };
    }
    return { type: ErrorTypes.NETWORK, message: 'Unable to connect to the prediction server.' };
  }

  const status = response.status;
  if (status === 404) {
    if (error.detail?.includes('Model')) {
      return { type: ErrorTypes.MODEL_NOT_FOUND, message: 'Selected model not found.' };
    }
    if (error.detail?.includes('Pool')) {
      return { type: ErrorTypes.POOL_NOT_FOUND, message: 'Pool not found on this network.' };
    }
    return { type: ErrorTypes.SERVER, message: error.detail || 'Resource not found.' };
  }
  if (status === 422) {
    return { type: ErrorTypes.VALIDATION, message: error.detail || 'Invalid request parameters.' };
  }
  if (status >= 500) {
    return { type: ErrorTypes.SERVER, message: 'Server error. Please try again later.' };
  }

  return { type: ErrorTypes.UNKNOWN, message: error.detail || 'An unexpected error occurred.' };
};

// Async thunk: Fetch available models
export const fetchAvailableModels = createAsyncThunk(
  'prediction/fetchAvailableModels',
  async (_, { rejectWithValue }) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/models/available`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const categorized = categorizeError(errorData, response);
        return rejectWithValue(categorized);
      }

      const data = await response.json();
      if (data.status === 'success' && data.models?.length > 0) {
        return data.models;
      }
      return rejectWithValue({ type: ErrorTypes.SERVER, message: 'No models available.' });
    } catch (error) {
      clearTimeout(timeoutId);
      const categorized = categorizeError(error, null);
      return rejectWithValue(categorized);
    }
  }
);

// Async thunk: Run backtest on test data
export const runBacktest = createAsyncThunk(
  'prediction/runBacktest',
  async ({ poolId, protocolId, modelName, nEpisodes = 1, episodeLengthDays = 18, debug = false }, { rejectWithValue }) => {
    console.log('[Redux] runBacktest called with params:', { poolId, protocolId, modelName, episodeLengthDays, debug });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout for backtest

    const requestBody = {
      pool_id: poolId,
      protocol_id: protocolId,
      model_name: modelName,
      n_episodes: nEpisodes,
      episode_length_days: episodeLengthDays,
      debug: debug
    };

    console.log('[Redux] Sending request to:', `${API_BASE_URL}/api/v1/backtest`);
    console.log('[Redux] Request body:', requestBody);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      console.log('[Redux] Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const categorized = categorizeError(errorData, response);
        return rejectWithValue(categorized);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      const categorized = categorizeError(error, null);
      return rejectWithValue(categorized);
    }
  }
);

// Async thunk: Get AI prediction
export const getPrediction = createAsyncThunk(
  'prediction/getPrediction',
  async ({ poolId, protocolId, modelName, investment }, { rejectWithValue }) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for prediction

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pool_id: poolId,
          protocol_id: protocolId,
          model_name: modelName,
          investment: investment || 10000
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const categorized = categorizeError(errorData, response);
        return rejectWithValue(categorized);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      const categorized = categorizeError(error, null);
      return rejectWithValue(categorized);
    }
  }
);

const initialState = {
  // Available models
  models: [],
  modelsLoading: false,
  modelsError: null,

  // Selected model
  selectedModel: null,

  // Prediction result
  prediction: null,
  predictionLoading: false,
  predictionError: null,

  // Backtest result
  backtest: null,
  backtestLoading: false,
  backtestError: null,
  backtestDays: 30,  // Backtest period in days (shared with StrategyBacktest)

  // Loading progress (for future use with chunked data)
  loadingProgress: {
    current: 0,
    total: 0,
    message: ''
  }
};

const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    setSelectedModel: (state, action) => {
      state.selectedModel = action.payload;
    },
    clearPrediction: (state) => {
      state.prediction = null;
      state.predictionError = null;
    },
    clearBacktest: (state) => {
      state.backtest = null;
      state.backtestError = null;
    },
    setBacktestDays: (state, action) => {
      state.backtestDays = action.payload;
    },
    clearErrors: (state) => {
      state.modelsError = null;
      state.predictionError = null;
      state.backtestError = null;
    },
    setLoadingProgress: (state, action) => {
      state.loadingProgress = action.payload;
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch available models
      .addCase(fetchAvailableModels.pending, (state) => {
        state.modelsLoading = true;
        state.modelsError = null;
      })
      .addCase(fetchAvailableModels.fulfilled, (state, action) => {
        state.modelsLoading = false;
        state.models = action.payload;
        // Auto-select first model if none selected
        if (!state.selectedModel && action.payload.length > 0) {
          state.selectedModel = action.payload[0].name;
        }
      })
      .addCase(fetchAvailableModels.rejected, (state, action) => {
        state.modelsLoading = false;
        state.modelsError = action.payload;
      })

      // Get prediction
      .addCase(getPrediction.pending, (state) => {
        state.predictionLoading = true;
        state.predictionError = null;
        state.loadingProgress = { current: 0, total: 100, message: 'Fetching pool data...' };
      })
      .addCase(getPrediction.fulfilled, (state, action) => {
        state.predictionLoading = false;
        state.prediction = action.payload;
        state.loadingProgress = { current: 100, total: 100, message: 'Complete!' };
      })
      .addCase(getPrediction.rejected, (state, action) => {
        state.predictionLoading = false;
        state.predictionError = action.payload;
        state.loadingProgress = { current: 0, total: 0, message: '' };
      })

      // Run backtest
      .addCase(runBacktest.pending, (state) => {
        state.backtestLoading = true;
        state.backtestError = null;
        state.loadingProgress = { current: 0, total: 100, message: 'Running backtest on test data...' };
      })
      .addCase(runBacktest.fulfilled, (state, action) => {
        state.backtestLoading = false;
        state.backtest = action.payload;
        state.loadingProgress = { current: 100, total: 100, message: 'Backtest complete!' };
      })
      .addCase(runBacktest.rejected, (state, action) => {
        state.backtestLoading = false;
        state.backtestError = action.payload;
        state.loadingProgress = { current: 0, total: 0, message: '' };
      });
  }
});

// Actions
export const {
  setSelectedModel,
  clearPrediction,
  clearBacktest,
  setBacktestDays,
  clearErrors,
  setLoadingProgress
} = predictionSlice.actions;

// Selectors
export const selectModels = (state) => state.prediction.models;
export const selectModelsLoading = (state) => state.prediction.modelsLoading;
export const selectModelsError = (state) => state.prediction.modelsError;
export const selectSelectedModel = (state) => state.prediction.selectedModel;
export const selectPrediction = (state) => state.prediction.prediction;
export const selectPredictionLoading = (state) => state.prediction.predictionLoading;
export const selectPredictionError = (state) => state.prediction.predictionError;
export const selectLoadingProgress = (state) => state.prediction.loadingProgress;
export const selectBacktest = (state) => state.prediction.backtest;
export const selectBacktestLoading = (state) => state.prediction.backtestLoading;
export const selectBacktestError = (state) => state.prediction.backtestError;
export const selectBacktestDays = (state) => state.prediction.backtestDays;

// Helper selector: Get selected model metadata
export const selectSelectedModelMetadata = (state) => {
  const models = state.prediction.models;
  const selectedName = state.prediction.selectedModel;
  return models.find(m => m.name === selectedName)?.metadata || null;
};

export default predictionSlice.reducer;
