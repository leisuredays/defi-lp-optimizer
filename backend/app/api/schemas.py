"""
API Request/Response Schemas using Pydantic

Defines data models for the ML optimization API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from datetime import datetime


class RangeConfig(BaseModel):
    """Price range configuration for a strategy"""
    min: float = Field(..., description="Minimum price of range", gt=0)
    max: float = Field(..., description="Maximum price of range", gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "min": 1400.0,
                "max": 1600.0
            }
        }


class OptimizeRequest(BaseModel):
    """Request payload for POST /api/v1/optimize endpoint"""
    pool_id: str = Field(..., description="Pool ID from The Graph")
    protocol_id: int = Field(..., description="Protocol ID (0=Ethereum, 1=Optimism, 2=Arbitrum, 3=Polygon, 4=Celo)")
    days_history: int = Field(default=30, description="Number of days of historical data to use", ge=7, le=365)
    current_ranges: Dict[str, RangeConfig] = Field(..., description="Current strategy ranges (S1, S2)")
    current_price: float = Field(..., description="Current market price", gt=0)
    investment: float = Field(..., description="Investment amount in USD", gt=0)
    fee_tier: int = Field(..., description="Pool fee tier (500, 3000, or 10000 basis points)")

    class Config:
        json_schema_extra = {
            "example": {
                "pool_id": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
                "protocol_id": 0,
                "days_history": 30,
                "current_ranges": {
                    "S1": {"min": 1400.0, "max": 1600.0},
                    "S2": {"min": 1300.0, "max": 1700.0}
                },
                "current_price": 1500.0,
                "investment": 10000.0,
                "fee_tier": 3000
            }
        }


class RangeRecommendation(BaseModel):
    """ML-recommended price range for a strategy"""
    min: float = Field(..., description="Recommended minimum price")
    max: float = Field(..., description="Recommended maximum price")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "min": 1420.0,
                "max": 1580.0,
                "confidence": 0.85
            }
        }


class ExpectedPerformance(BaseModel):
    """Expected performance metrics for recommended ranges"""
    apr: float = Field(..., description="Expected Annual Percentage Return")
    expected_fees: float = Field(..., description="Expected fees earned (USD)")
    expected_il: float = Field(..., description="Expected impermanent loss (USD)")
    gas_costs: float = Field(..., description="Estimated gas costs (USD)")
    net_return: float = Field(..., description="Net return after fees, IL, and gas (USD)")

    class Config:
        json_schema_extra = {
            "example": {
                "apr": 45.5,
                "expected_fees": 1250.0,
                "expected_il": 80.0,
                "gas_costs": 20.0,
                "net_return": 1150.0
            }
        }


class OptimizeResponse(BaseModel):
    """Response payload for POST /api/v1/optimize endpoint"""
    status: str = Field(..., description="Response status (success or error)")
    recommendations: Dict[str, RangeRecommendation] = Field(..., description="Recommended ranges by strategy ID")
    expected_performance: ExpectedPerformance = Field(..., description="Expected performance metrics")
    model_version: str = Field(..., description="ML model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "recommendations": {
                    "S1": {
                        "min": 1420.0,
                        "max": 1580.0,
                        "confidence": 0.85
                    },
                    "S2": {
                        "min": 1350.0,
                        "max": 1650.0,
                        "confidence": 0.82
                    }
                },
                "expected_performance": {
                    "apr": 45.5,
                    "expected_fees": 1250.0,
                    "expected_il": 80.0,
                    "gas_costs": 20.0,
                    "net_return": 1150.0
                },
                "model_version": "v1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ModelStatusResponse(BaseModel):
    """Response payload for GET /api/v1/model/status endpoint"""
    model_version: str = Field(..., description="Current model version")
    model_path: str = Field(..., description="Path to model file")
    trained_on_pools: int = Field(..., description="Number of pools used for training")
    training_date: Optional[str] = Field(None, description="Date when model was trained")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    status: str = Field(..., description="Model status (ready, training, error)")

    class Config:
        json_schema_extra = {
            "example": {
                "model_version": "v1.0.0",
                "model_path": "models/ppo_uniswap_v3_v1.0.0.zip",
                "trained_on_pools": 20,
                "training_date": "2024-01-10",
                "performance_metrics": {
                    "win_rate": 0.72,
                    "avg_apr_improvement": 18.5,
                    "sharpe_ratio": 1.65
                },
                "status": "ready"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response payload for GET /api/v1/health endpoint"""
    status: str = Field(..., description="Health status (healthy or unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class PredictRequest(BaseModel):
    """Request payload for POST /api/v1/predict endpoint"""
    pool_id: str = Field(..., description="Pool ID from The Graph")
    protocol_id: int = Field(..., description="Protocol ID (0=Ethereum, 1=Optimism, 2=Arbitrum, 3=Polygon, 5=Celo)")
    model_name: str = Field(default="ppo_arbitrum_usdc_weth_005", description="Model name to use for prediction")
    investment: float = Field(default=10000, description="Investment amount in USD", gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "pool_id": "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",
                "protocol_id": 2,
                "model_name": "ppo_arbitrum_usdc_weth_005",
                "investment": 10000.0
            }
        }


class PredictResponse(BaseModel):
    """Response payload for POST /api/v1/predict endpoint"""
    status: str = Field(..., description="Response status (success or error)")
    pool_info: Dict[str, Any] = Field(..., description="Pool information")
    current_state: Dict[str, float] = Field(..., description="Current market state features")
    predicted_range: RangeRecommendation = Field(..., description="AI-predicted optimal range")
    model_name: str = Field(..., description="Model name used")
    model_version: str = Field(..., description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "pool_info": {
                    "pool_id": "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",
                    "token0": "WETH",
                    "token1": "USDC",
                    "fee_tier": 500,
                    "current_price": 0.00035
                },
                "current_state": {
                    "price": 0.00035,
                    "volatility": 0.05,
                    "liquidity": 1500000
                },
                "predicted_range": {
                    "min": 0.00025,
                    "max": 0.00045,
                    "confidence": 0.92
                },
                "model_name": "ppo_arbitrum_usdc_weth_005",
                "model_version": "v1.0.0",
                "timestamp": "2024-12-20T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response payload"""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Failed to fetch pool data",
                "detail": "The Graph API returned 404: Pool not found",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
